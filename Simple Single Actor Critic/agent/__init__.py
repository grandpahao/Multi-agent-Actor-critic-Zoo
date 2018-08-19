from .actor import Actor
from .critic import Critic


def get_agent(agent_type, **kwargs):
    if agent_type == 'actor':
        agent = Actor(kwargs['n_ac'], kwargs['lr'], kwargs['test'])
    elif agent_type == 'critic':
        agent = Critic(kwargs['lr'], kwargs['discount'], kwargs['test'])
    else:
        raise Exception('{} is not supported!'.format(agent_type))

    return agent
