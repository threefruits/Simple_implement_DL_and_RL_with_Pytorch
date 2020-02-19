import numpy as np
import pyglet

class ArmEnv(object):
    viewer = None       # 首先没有 viewer
    dt = 0.05  # 转动的速度和 dt 有关
    action_bound = [-1, 1]  # 转动的角度范围
    goal = {'x': 150., 'y': 150., 'l': 40}  # 蓝色 goal 的 x,y 坐标和长度 l
    state_dim = 13 # 两个观测值
    action_dim = 3

    def __init__(self):
        self.arm_info = np.zeros(
            3, dtype=[('l', np.float32), ('r', np.float32)])
        # 生成出 (2,2) 的矩阵
        self.arm_info['l'] = 200  # 两段手臂都 100 长
        self.arm_info['r'] = np.pi / 6  # 两段手臂的端点角度

    def render(self):
            if self.viewer is None: # 如果调用了 render, 而且没有 viewer, 就生成一个
                self.viewer = Viewer(self.arm_info,self.goal)
            self.viewer.render()    # 使用 Viewer 中的 render 功能

    def step(self, action):
        done = False
        r = 0.

        # 计算单位时间 dt 内旋转的角度, 将角度限制在360度以内
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2  # normalize

        # 我们可以将两截手臂的角度信息当做一个 state (之后会变)

        # 如果手指接触到蓝色的 goal, 我们判定结束回合 (done)
        # 所以需要计算 finger 的坐标
        (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([400., 400.])  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_
        finger = np.array([np.cos(a1r + a2r+ a3r), np.sin(a1r + a2r+ a3r)]) * a3l + a2xy_  # a2 end (x2, y2)
        dist0 = [(self.goal['x'] - a1xy_[0]) / 800, (self.goal['y'] - a1xy_[1]) / 800]
        dist1 = [(self.goal['x'] - a2xy_[0]) / 800, (self.goal['y'] - a2xy_[1]) / 800]
        dist2 = [(self.goal['x'] - finger[0]) / 800, (self.goal['y'] - finger[1]) / 800]
        r = -np.sqrt(dist2[0] ** 2 + dist2[1] ** 2)
        # 根据 finger 和 goal 的坐标得出 done and reward
        if self.goal['x'] - self.goal['l']/2 < finger[0] < self.goal['x'] + self.goal['l']/2:
            if self.goal['y'] - self.goal['l']/2 < finger[1] < self.goal['y'] + self.goal['l']/2:
                r += 0.5
                self.on_goal += 1
                if self.on_goal > 50:
                    done = True
        else:
            self.on_goal = 0
        # s = self.arm_info['r']
        s = np.concatenate((a1xy_ / 400, a2xy_ /400 ,finger / 400, dist0 + dist1 + dist2,[1. if self.on_goal else 0.]))
        return s, r, done

    def reset(self):
        self.arm_info['r'] = 2 * np.pi * np.random.rand(3)
        self.goal['x'] = np.random.rand() * 400.
        self.goal['y'] = np.random.rand() * 400.
        self.on_goal = 0
        (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([400., 400.])  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_
        finger = np.array([np.cos(a1r + a2r + a3r), np.sin(a1r + a2r + a3r)]) * a3l + a2xy_  # a2 end (x2, y2)
        dist0 = [(self.goal['x'] - a1xy_[0]) / 800, (self.goal['y'] - a1xy_[1]) / 800]
        dist1 = [(self.goal['x'] - a2xy_[0]) / 800, (self.goal['y'] - a2xy_[1]) / 800]
        dist2 = [(self.goal['x'] - finger[0]) / 800, (self.goal['y'] - finger[1]) / 800]
        s = np.concatenate((a1xy_ / 400, a2xy_ / 400, finger / 400, dist0 + dist1 + dist2, [1. if self.on_goal else 0.]))
        return s
    def sample_action(self):
        return np.random.rand(3) - 0.5  # two radians

class Viewer(pyglet.window.Window):
    bar_thc = 3
    def __init__(self,arm_info,goal):
        super(Viewer, self).__init__(width=800, height=800, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.batch = pyglet.graphics.Batch()
        self.goal_info = goal
        self.arm_info = arm_info
        # 添加窗口中心点, 手臂的根
        self.center_coord = np.array([400, 400])

        ...
        # 蓝色 goal 的信息包括他的 x, y 坐标, goal 的长度 l
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))  # color


        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,  # 同上, 点信息
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))  # color

        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,  # 同上, 点信息
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))  # color

        self.arm3 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,  # 同上, 点信息
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))  # color
    def render(self):
        self._update_arm()  # 更新手臂内容 (暂时没有变化)
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()
    def on_draw(self):
        self.clear()  # 清屏
        self.batch.draw()  # 画上 batch 里面的内容

    def _update_arm(self):
        self.goal.vertices = (
            self.goal_info['x'] - self.goal_info['l'] / 2, self.goal_info['y'] - self.goal_info['l'] / 2,
            self.goal_info['x'] + self.goal_info['l'] / 2, self.goal_info['y'] - self.goal_info['l'] / 2,
            self.goal_info['x'] + self.goal_info['l'] / 2, self.goal_info['y'] + self.goal_info['l'] / 2,
            self.goal_info['x'] - self.goal_info['l'] / 2, self.goal_info['y'] + self.goal_info['l'] / 2)

        (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']  # radian, angle
        a1xy = self.center_coord  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        a3xy_ = np.array([np.cos(a1r + a2r + a3r), np.sin(a1r + a2r + a3r)]) * a3l + a2xy_
        # 第一段手臂的4个点信息
        a1tr,a2tr,a3tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'][0] - self.arm_info['r'][1], np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        # 第二段手臂的4个点信息
        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        xy21_ = a2xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc
        xy22_ = a2xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc
        xy31 = a3xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc
        xy32 = a3xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc

        # 将点信息都放入手臂显示中
        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))
        self.arm3.vertices = np.concatenate((xy21_, xy22_, xy31, xy32))

    def on_mouse_motion(self, x, y, dx, dy):
        self.goal_info['x'] = x
        self.goal_info['y'] = y


if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.render()
        env.step(env.sample_action())