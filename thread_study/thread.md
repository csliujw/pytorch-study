# 多线程笔记
- thread模块，没有
- threading模块有

## 全局解释器锁（GIL）
全局解释器相关信息
- GIL Global Interpreter Lock: 全局解释器锁
- 主循环中同时只能有一个控制线程在执行。
- 尽管Python解释器中可运行多个线程，但是在任意给定的时刻只有一个线程会被解释器执行。
线程运行方式
- 设置GIL
- 切换进一个线程去运行
- 执行下面操作之一
    - 指定数量的字节码指令
    - 线程主动让出控制权
- 把线程设置会睡眠状态（切换出线程）
- 解锁GIL
- 重复上述步骤
IO密集型的Python程序要比计算密集型的代码更好地利用多线程环境

## threading模块
### threading中的相关对象
- Thread : 执行线程的对象
- Lock : 锁
- RLock : 可重入锁
- Condition : 条件变量
- Event : 条件变量通用版，任意数量线程等地事件的发生，在该事件发生后所有线程将被激活
- Semaphore : 信号量
- BoundedSemaphore : 不允许超过初值的信号量
- Timer : 
- Barrier : 屏障，到达指定数量线程后才可进行往下运行。
### thread类
- 创建线程的三种方式
    - 创建Thread的实例，传给它一个函数
    - 创建Thread的实例，传给它一个可调用的类实例
    - 派生Thread的子类，并创建子类的实例
- 仅看构造方法
```python
# group 预留给将来扩展ThreadGroup时使用类实现。
# target 就是这个线程中要运行的方法
def __init__(self, group=None, target=None, name=None,
             args=(), kwargs=None, *, daemon=None):
    """This constructor should always be called with keyword arguments. Arguments are:

    *group* should be None; reserved for future extension when a ThreadGroup
    class is implemented.

    *target* is the callable object to be invoked by the run()
    method. Defaults to None, meaning nothing is called.

    *name* is the thread name. By default, a unique name is constructed of
    the form "Thread-N" where N is a small decimal number.

    *args* is the argument tuple for the target invocation. Defaults to ().

    *kwargs* is a dictionary of keyword arguments for the target
    invocation. Defaults to {}.

    If a subclass overrides the constructor, it must make sure to invoke
    the base class constructor (Thread.__init__()) before doing anything
    else to the thread.

    """
```