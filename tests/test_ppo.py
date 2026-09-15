import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from agents.ppo_agent import PPOAgent, clipped_policy_loss
from agents.dqn_agent import DQNAgent
from models.actor_critic import ActorCritic
from train_ppo import Collector, train
from utils.evaluation import load_policy, evaluate_policy
from utils.ppo_support import action_mask, reward_components, rng_state, restore_rng
from utils.rollout_buffer import compute_gae
from utils.seeding import set_seed
from utils.strategy import strategy_score, strategy_reward
from env.game_2048 import Game2048


def config():
    c=json.loads((Path(__file__).parents[1]/'configs/ppo_smoke.json').read_text())
    c.update(num_envs=2,rollout_steps=4,minibatch_size=4,update_epochs=2,total_steps=8,
             validation_interval=8,validation_seeds=[100000],max_episode_steps=3)
    return c


class PPOTests(unittest.TestCase):
    def setUp(self):
        set_seed(7)

    def test_mask_and_single_action(self):
        model=ActorCritic()
        masks=torch.tensor([[0,0,1,0],[1,0,1,0]],dtype=torch.bool)
        dist,_=model.distribution(torch.zeros(2,16),masks)
        self.assertTrue((dist.probs[~masks]==0).all())
        self.assertEqual(dist.probs[0,2],1)
        self.assertEqual(dist.entropy()[0],0)
        for _ in range(20):
            self.assertEqual(dist.sample()[0],2)
        with self.assertRaises(ValueError):
            model.distribution(torch.zeros(1,16),torch.zeros(1,4,dtype=torch.bool))
        env=Game2048(); env.reset()
        self.assertEqual(np.flatnonzero(action_mask(env)).tolist(),env.get_valid_actions())

    def test_ratio_and_update_saved_mask(self):
        c=config(); agent=PPOAgent(c); collector=Collector(c)
        data,_,_=collector.collect(agent,17)
        self.assertEqual(len(data['action']),17)
        for name in ('old_log_prob','old_value','returns','advantages'):
            self.assertFalse(data[name].requires_grad)
        for env in collector.envs:
            env.board.fill(0)  # Current boards cannot affect historical likelihoods.
        dist,_=agent.model.distribution(data['state'],data['mask'])
        torch.testing.assert_close((dist.log_prob(data['action'])-data['old_log_prob']).exp(),torch.ones(17))
        before=[p.detach().clone() for p in agent.model.parameters()]
        metrics=agent.update(data)
        self.assertLess(metrics['initial_ratio_max_error'],1e-5)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,agent.model.parameters())))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in agent.model.parameters()))

    def test_clipping_both_signs(self):
        r=torch.tensor([1.5,.5,1.5,.5],requires_grad=True)
        a=torch.tensor([2.,2.,-2.,-2.])
        loss=clipped_policy_loss(r,a,.2)
        self.assertAlmostEqual(loss.item(),.3,places=6) # -(2.4+1-3-1.6)/4
        loss.backward()
        torch.testing.assert_close(r.grad,torch.tensor([0.,-.5,.5,0.]))

    def test_hand_gae_terminal_and_rollout_cut(self):
        adv,ret=compute_gae([1,2],[.5,.6],[.6,99],[False,True],[False,False],.9,1.)
        np.testing.assert_allclose(adv,[2.3,1.4],rtol=1e-6)
        np.testing.assert_allclose(ret,[2.8,2.],rtol=1e-6)
        adv,_=compute_gae([1],[.5],[2],[False],[False],.9,.95)
        np.testing.assert_allclose(adv,[2.3])

    def test_truncation_bootstraps_but_breaks_trace(self):
        adv,_=compute_gae([1,100],[.5,0],[2,0],[False,True],[True,False],.9,1)
        np.testing.assert_allclose(adv,[2.3,100])

    def test_multiple_environment_episode_boundaries(self):
        adv,_=compute_gae([[1,10],[2,20]],[[0,0],[0,0]],[[0,0],[0,0]],
                          [[True,False],[True,True]],[[False,False],[False,False]],1,1)
        np.testing.assert_allclose(adv,[[1,30],[2,20]])
        c=config(); collector=Collector(c); agent=PPOAgent(c)
        data,episodes,_=collector.collect(agent,13)
        self.assertEqual(len(data['action']),13)
        self.assertEqual(len(episodes),4)
        self.assertTrue(all(e['length']==3 and e['truncated'] for e in episodes))

    def test_reward_components_and_accounting(self):
        c=config(); env=Game2048(); env.reset(); before=env.board.copy()
        _,raw,term,_=env.step(env.get_valid_actions()[0])
        for mode in ('raw','potential','legacy'):
            c['reward_mode']=mode
            r,s,t=reward_components(raw,before,env.board,term,c)
            self.assertAlmostEqual(t,(r+s)*c['reward_scale'])
            expected=0 if mode=='raw' else (strategy_reward(before,env.board) if mode=='legacy' else
                     c['shaping_coef']*(c['gamma']*strategy_score(env.board)-strategy_score(before)))
            self.assertAlmostEqual(s,expected)
        _,s,_=reward_components(0,before,env.board,True,c)
        self.assertEqual(s,strategy_reward(before,env.board))
        c['reward_mode']='potential'
        _,s,_=reward_components(0,before,env.board,True,c)
        self.assertEqual(s,-c['shaping_coef']*strategy_score(before))
        collector=Collector(c); _,episodes,rewards=collector.collect(PPOAgent(c),12)
        self.assertAlmostEqual(sum(e['raw_reward'] for e in episodes),rewards[0])
        self.assertTrue(all(e['score']==e['raw_reward'] for e in episodes))
        self.assertAlmostEqual(rewards[2],c['reward_scale']*(rewards[0]+rewards[1]))

    def test_validation_preserves_rng_and_collector(self):
        c=config(); agent=PPOAgent(c); collector=Collector(c)
        state=copy.deepcopy(collector.state_dict()); rng=rng_state()
        expected=(torch.rand(3),np.random.rand(),__import__('random').random())
        restore_rng(rng)
        evaluate_policy(agent,[200000])
        torch.testing.assert_close(expected[0],torch.rand(3),rtol=0,atol=0)
        self.assertEqual(expected[1],np.random.rand())
        self.assertEqual(expected[2],__import__('random').random())
        for a,b in zip(state['envs'],collector.state_dict()['envs']):
            np.testing.assert_array_equal(a['board'],b['board'])
            self.assertEqual(a['rng'],b['rng'])

    def test_checkpoint_dispatch_and_exact_resume(self):
        c=config()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            train(c,root/'first')
            c['total_steps']=16
            resumed=train(c,root/'resume',root/'first/checkpoints/last.pt')
            full=train(c,root/'full')
            for a,b in zip(resumed.model.parameters(),full.model.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            loaded,meta=load_policy(root/'resume/checkpoints/last.pt','ppo')
            self.assertEqual(meta['steps'],16)
            self.assertEqual(meta['run_steps'],16)
            with self.assertRaises(ValueError):
                load_policy(root/'resume/checkpoints/last.pt','dqn')
            dqn=DQNAgent(); torch.save(dqn.online_network.state_dict(),root/'old.pt')
            self.assertIsInstance(load_policy(root/'old.pt')[0],DQNAgent)
            c['reward_mode']='potential'
            with self.assertRaises(ValueError):
                train(c,root/'bad',root/'first/checkpoints/last.pt')

    def test_collector_bootstraps_final_observation_before_reset(self):
        c=config(); c.update(num_envs=1,max_episode_steps=1)
        agent=PPOAgent(c); collector=Collector(c)
        snapshot=collector.state_dict()
        data,episodes,_=collector.collect(agent,1)
        reference=Game2048()
        reference.board=snapshot['envs'][0]['board'].copy()
        reference.score=snapshot['envs'][0]['score']
        reference.rng.bit_generator.state=snapshot['envs'][0]['rng']
        final,_,_,_=reference.step(int(data['action'][0]))
        with torch.no_grad():
            expected=agent.model(torch.as_tensor(final))[1]
        torch.testing.assert_close(data['next_value'][0],expected)
        self.assertTrue(episodes[0]['truncated'])
        self.assertFalse(np.array_equal(final,collector.envs[0].get_state()))

    def test_kl_early_stopping(self):
        c=config(); c.update(target_kl=1e-10,update_epochs=8,learning_rate=.01)
        agent=PPOAgent(c)
        data,_,_=Collector(c).collect(agent,32)
        metrics=agent.update(data)
        self.assertTrue(metrics['kl_early_stop'])
        self.assertGreater(metrics['stopping_kl'],c['target_kl'])
        self.assertLess(metrics['optimizer_updates'],64)

    def test_one_env_partial_budget(self):
        c=config(); c['num_envs']=1
        data,_,_=Collector(c).collect(PPOAgent(c),5)
        self.assertEqual(len(data['action']),5)


if __name__=='__main__':
    unittest.main()
