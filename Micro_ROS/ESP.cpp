#include <micro_ros_arduino.h>
#include <geometry_msgs/msg/twist.h>
#include <nav_msgs/msg/odometry.h>

// --- PID Constants for Motors ---
float Kp = 0.5, Ki = 0.05, Kd = 0.1;
float current_battery = 100.0;

// ROS 2 Objects
rcl_publisher_t consensus_pub;
geometry_msgs__msg__Twist msg;

void loop() {
    // 1. Read Sensors (Distance & Battery)
    float front_dist = read_sonar();
    current_battery -= 0.01; 

    // 2. APF Logic (Repulsion from obstacles)
    float repulsive_x = 0, repulsive_y = 0;
    if (front_dist < 0.2) { // 20cm threshold
        repulsive_x = -1.0 / (front_dist * front_dist);
    }

    // 3. Social Consensus (Received from other robots via WiFi)
    float target_x = swarm_consensus.x;
    float target_y = swarm_consensus.y;

    // 4. Calculate Final Velocity (The "NeuroBot" Brain)
    msg.linear.x = (target_x + repulsive_x) * Kp;
    msg.angular.z = (target_y) * Kd;

    // 5. Emergency Power Mode
    if (current_battery < 20.0) {
        return_to_base();
    }

    // 6. Execute via PWM to Motors
    drive_motors(msg.linear.x, msg.angular.z);
    
    // 7. Publish state to the rest of the swarm
    rcl_publish(&consensus_pub, &msg, NULL);
    
    delay(100);
}
