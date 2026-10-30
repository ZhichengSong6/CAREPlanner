"""Explicit benchmark overrides while retaining Gazebo's display connection."""
import os

RENDER_KEYS = ('DISPLAY', 'XAUTHORITY', 'XDG_RUNTIME_DIR')


def simulation_environment(settings, inherited=None):
    inherited = os.environ if inherited is None else inherited
    env = dict(HOME=inherited.get('HOME', os.path.expanduser('~')),
               USER=inherited.get('USER', 'zhicheng'),
               PATH='/usr/local/bin:/usr/bin:/bin', LANG='C.UTF-8',
               CONDA_SHLVL='0', ROS_MASTER_URI='http://localhost:11311')
    env.update({key: inherited[key] for key in RENDER_KEYS if inherited.get(key)})
    env.update(settings)
    if not env.get('DISPLAY', '').strip():
        raise RuntimeError('Gazebo depth cameras require the session DISPLAY; '
                           'GUI=false does not disable camera rendering requirements')
    return env
