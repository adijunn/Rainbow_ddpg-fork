# ddpg.py:
Equivalent of ddpg_learner.py in openai/baselines repository. I don't think anything needs to be changed here for now in terms of compatibility.

# main.py
Equivalent of run.py in openai/baselines repository. Things to note:
1. Add a get_env_type method and so that we can use our gym_cloth environment
2. The train method (in baselines-fork) and the run method are equivalent (starts the training procedure)
3. Need to figure out equivalent of build_env method in baselines-fork. Includes a make_vec_env function which I need to figure out as well (should just vectorize the environment?) 





# Rainbow DDPG Fork

This repository contains Rainbow DDPG algorithm from paper Sim-to-Real Reinforcement Learning for Deformable Object Manipulation along with a toy pushing task to demonstrate how to use the code.

## Instructions

The code was tested on Mac OS with Python3.6. Use of virtualenvs is recommended. To run:

```
pip install -r requirements.txt
python main.py
```

Runnign a full training may take more than 24 hours on a machine with Nvidia Titan GPU and use a considerable amount of memory.

To run a demonstration of the toy task:

```
pip install -r requirements.txt
python run_demo.py
```

Please note that the hyper parameters are not necessarily optimised for the task.



## References

For a complete list of references, please see the accompanying paper.

The learning algorithm is based on OpenAI baselines (https://github.com/openai/baselines), the perlin noise file is heavily based on https://github.com/nikagra/python-noise/blob/master/noise.py and robot meshes are generated from https://github.com/Kinovarobotics/kinova-ros. 
