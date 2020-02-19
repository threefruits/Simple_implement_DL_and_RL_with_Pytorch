import gym
import time
env = gym.make('Pendulum-v0')   #创造环境 #初始化环境，observation为环境状态
count = 0
print(env.action_space) # 查看这个环境中可用的 action 有多少个
print(env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个
print(env.observation_space.high)   # 查看 observation 最高取值
print(env.observation_space.low)    # 查看 observation 最低取值
for t in range(10):
    observation = env.reset()
    for i in range(500):
        # action = env.action_space.sample()  #随机采样动作
        if (observation[2]>0 ):
            action = [2]
        else:
            action = [-2]
        observation, reward, done, info = env.step(action)  #与环境交互，获得下一步的时刻
        env.render()         #绘制场景
        count+=1
        time.sleep(0.01)      #每次等待0.2s
        if done:
             break
        print(done)
    print(count)             #打印该次尝试的步数