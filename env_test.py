import simpy
def UAV_process(env, battery, latency):
    while True:
        yield env.timeout(1)  # 每1时间步更新状态
        battery -= 5  # 模拟能耗
        latency = max(0, latency - 10)  # 模拟网络延迟变化
env = simpy.Environment()
env.process(UAV_process(env, battery=100, latency=200))
env.run(until=100)  # 运行100时间步