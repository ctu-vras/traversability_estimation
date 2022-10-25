from __future__ import absolute_import, division, print_function
import rospy
from threading import Event

__all__ = ['TopicServiceProxy']


class TopicServiceProxy(object):
    """Service proxy wrapper around set of input and output topics."""

    def __init__(self, request, response, queue_size=2, timeout=None, return_incomplete=False):
        """Create a service proxy.

        Parameters
        - request: A (topic, type) tuple or a list of those.
        - response: A (topic, type) tuple or a list of those.
        - queue_size: Queue size for publishers and subscribers.
        """
        assert request is not None
        assert response is not None
        assert len(request) > 0
        assert len(response) > 0

        if isinstance(request[0], str):
            request = [request]
        if isinstance(response[0], str):
            response = [response]

        self.event = Event()
        self.timeout = timeout
        self.pubs = len(request) * [None]
        self.subs = len(response) * [None]
        self.response = len(response) * [None]
        for i, (topic, type) in enumerate(request):
            self.pubs[i] = rospy.Publisher(topic, type, queue_size=queue_size)
        for i, (topic, type) in enumerate(response):
            self.subs[i] = rospy.Subscriber(topic, type, lambda msg, i=i: self.callback(msg, i), queue_size=queue_size)

    def clear_response(self):
        self.response = len(self.response) * [None]

    def response_empty(self):
        return all([msg is None for msg in self.response])

    def response_complete(self):
        return all([msg is not None for msg in self.response])

    def callback(self, msg, i):
        assert self.response[i] is None
        self.response[i] = msg
        if self.response_complete():
            self.event.set()

    def call(self, msgs):
        """Call the service.

        Raises TimeoutError if response messages do not arrive in time.
        """
        assert len(msgs) == len(self.pubs)
        assert self.response_empty()
        assert not self.event.is_set()
        self.event.clear()
        for i, msg in enumerate(msgs):
            self.pubs[i].publish(msg)
        if not self.event.wait(self.timeout):
            raise TimeoutError('Service call timed out.')
        assert self.response_complete()
        response = self.response
        self.clear_response()
        return response

    def __call__(self, msgs):
        return self.call(msgs)


def test():
    import roslaunch
    from std_msgs.msg import String
    from time import sleep

    class RosCore(object):
        def __init__(self):
            uuid = roslaunch.rlutil.get_or_generate_uuid(options_runid=None, options_wait_for_master=False)
            roslaunch.configure_logging(uuid)
            self.launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_files=[], is_core=True)
            self.launch.start()

        def __del__(self):
            self.launch.shutdown()

    class Repeater(object):
        def __init__(self, input, output):
            self.pub = rospy.Publisher(output, String, queue_size=1)
            self.sub = rospy.Subscriber(input, String, self.callback, queue_size=1)

        def callback(self, msg):
            self.pub.publish(msg)

    roscore = RosCore()
    sleep(2)

    rospy.init_node('topic_service_proxy_test')

    print('Creating repeater service...')
    repeater = Repeater('request', 'response')
    sleep(2)

    print('Creating service proxy...')
    repeater_proxy = TopicServiceProxy([('request', String)], [('response', String)])
    void_proxy = TopicServiceProxy([('void', String)], [('silent', String)], timeout=1.0)
    sleep(2)

    request = String('Hello World!')

    try:
        print('Calling repeater service...')
        response, = repeater_proxy([request])
        assert response == request

        print('Calling void service...')
        response, = void_proxy([request])
        timed_out = False

    except TimeoutError as ex:
        timed_out = True
        print('Timed out.')

    assert timed_out


def main():
    test()


if __name__ == '__main__':
    main()
