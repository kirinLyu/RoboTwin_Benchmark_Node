import itertools
import numpy as np
import pytest
from forge_msgs import Image, JointCommand, JointState, PolicyCommand
from g05_node import G05PolicyNode, OBS_IDS


class Backend:
    def __init__(self):
        self.resets=0
        self.observations=[]
        self.inferences=0

    def reset(self):self.resets+=1
    def update_obs(self,obs):self.observations.append(obs)
    def get_action(self):
        self.inferences+=1
        return [{"left_arm_joint_state":np.full(6,n),"left_ee_joint_state":[n],
                 "right_arm_joint_state":np.full(6,n),"right_ee_joint_state":[n]} for n in (1,2)]


class Node:
    def __init__(self):self.actions=[]
    def send_output(self,channel,batch):
        assert channel=='action'
        self.actions.append(JointCommand.from_arrow(batch))


def command(policy,node,name):
    value=PolicyCommand.from_inputs(policy_id='robotwin_policy',command=name,inputs={'instruction':'adjust bottle'}).to_arrow()
    policy.process_event(node,{'type':'INPUT','id':'policy_command','value':value})


def frame(policy,node,channels):
    for channel in channels:
        value=(JointState.from_np(np.zeros(14),[f'j{i}' for i in range(14)],'position') if channel=='proprio_state'
               else Image.from_numpy(np.zeros((8,8,3),dtype=np.uint8),'rgb8')).to_arrow()
        policy.process_event(node,{'type':'INPUT','id':channel,'value':value})


@pytest.mark.parametrize('channels',list(itertools.permutations(sorted(OBS_IDS))))
def test_complete_frame_orders_and_official_chunk_cadence(channels):
    backend,node=Backend(),Node()
    policy=G05PolicyNode({'policy_id':'robotwin_policy'},backend)
    for name in ('start','set_instruction','reset_scene'):command(policy,node,name)
    for _ in range(3):frame(policy,node,channels)
    assert len(node.actions)==3
    assert len(backend.observations)==3
    assert backend.inferences==2


def test_reset_can_follow_new_frame_and_stop_allows_second_session():
    backend,node=Backend(),Node()
    policy=G05PolicyNode({'policy_id':'robotwin_policy'},backend)
    for _ in range(2):
        for name in ('start','set_instruction'):command(policy,node,name)
        before=len(node.actions)
        frame(policy,node,sorted(OBS_IDS))
        assert len(node.actions)==before
        command(policy,node,'reset_scene')
        assert len(node.actions)==before+1
        command(policy,node,'stop')
        assert not policy.running and not policy.chunk and not policy.observations
    assert backend.inferences==2
    assert not policy.process_event(node,{'type':'STOP'})
