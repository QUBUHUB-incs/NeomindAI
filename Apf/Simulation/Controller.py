import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PID:
    """Standard PID Controller for smooth force-to-motor translation."""
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_error = np.zeros(2)
        self.integral = np.zeros(2)

    def compute(self, target_vel, current_vel, dt=0.1):
        error = target_vel - current_vel
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

class SwarmRobot:
    def __init__(self, robot_id, start_pos):
        self.id = robot_id
        self.pos = np.array(start_pos, dtype=float)
        self.vel = np.zeros(2)
        self.belief_goal = np.array([80.0, 80.0]) # Shared swarm target
        self.pid = PID(kp=0.6, ki=0.01, kd=0.1) # Smoother movement
        
        # Behavior Constants
        self.max_speed = 3.0
        self.safe_dist = 6.0
        self.view_dist = 15.0

    def compute_forces(self, obstacles, neighbors):
        # 1. APF: Attraction to goal & Repulsion from obstacles
        f_att = 0.4 * (self.belief_goal - self.pos)
        f_rep = np.zeros(2)
        for obs in obstacles:
            dist = np.linalg.norm(self.pos - obs)
            if dist < self.view_dist:
                f_rep += 15.0 * (1.0/dist - 1.0/self.view_dist) * (self.pos - obs) / dist**3

        # 2. Boids: Separation & Alignment
        f_sep, f_ali = np.zeros(2), np.zeros(2)
        if neighbors:
            for n in neighbors:
                d = np.linalg.norm(self.pos - n.pos)
                if d < self.safe_dist:
                    f_sep += (self.pos - n.pos) / (d**2)
            f_ali = 0.1 * (np.mean([n.vel for n in neighbors], axis=0) - self.vel)

        # 3. Consensus Update
        if neighbors:
            avg_belief = np.mean([n.belief_goal for n in neighbors], axis=0)
            self.belief_goal = 0.95 * self.belief_goal + 0.05 * avg_belief

        return f_att + f_rep + f_sep + f_ali

    def step(self, obstacles, neighbors):
        target_force = self.compute_forces(obstacles, neighbors)
        # Apply PID to target force for "drive" feel
        acceleration = self.pid.compute(target_force, self.vel)
        self.vel += acceleration
        
        # Clamp Speed
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed
        self.pos += self.vel

# --- Visualization Setup ---
fig, ax = plt.subplots(figsize=(8, 8))
robots = [SwarmRobot(i, np.random.rand(2)*40) for i in range(12)]
obstacles = [np.array([40, 40]), np.array([20, 60]), np.array([60, 20])]
scat = ax.scatter([r.pos[0] for r in robots], [r.pos[1] for r in robots], c='blue', s=100, label='Robots')
ax.scatter([o[0] for o in obstacles], [o[1] for o in obstacles], c='red', marker='X', s=200, label='Obstacles')
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.legend()

def update(frame):
    for r in robots:
        neighbors = [n for n in robots if n.id != r.id and np.linalg.norm(r.pos - n.pos) < 25]
        r.step(obstacles, neighbors)
    
    scat.set_offsets([r.pos for r in robots])
    return scat,

ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()
