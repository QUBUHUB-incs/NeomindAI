import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import paho.mqtt.client as mqtt
import json

MQTT_BROKER = "127.0.0.1:80001"
client = mqtt.Client("neurobot01")
client.connect(MQTT_BROKER)

class SwarmNode(Node):
    def __init__(self):
        super().__init__('swarm_node')
        self.create_subscription(LaserScan, 'lidar', self.lidar_callback, 10)

    def lidar_callback(self, msg):
        # Publish sensor info to swarm
        data = {"lidar": msg.ranges}
        client.publish("neurobot/swarm", json.dumps(data))

def main(args=None):
    rclpy.init(args=args)
    node = SwarmNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
