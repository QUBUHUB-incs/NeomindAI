import numpy as np

class SwarmRobot:
    def __init__(self, robot_id, start_pos):
        self.id = robot_id
        self.pos = np.array(start_pos, dtype=float)
        self.vel = np.zeros(2)
        self.belief_goal = np.array([100.0, 100.0]) # Initial "opinion" of goal
        
        # Physics Parameters
        self.max_speed = 2.0
        self.influence_dist = 15.0 # How far it "sees" obstacles
        self.safe_dist = 5.0      # Separation distance from peers

    def compute_forces(self, global_goal, obstacles, neighbors):
        """
        1. APF: Attraction to Goal & Repulsion from Obstacles
        2. Boids: Alignment, Cohesion, Separation
        3. Consensus: Averaging the Goal Belief
        """
        # --- APF NAVIGATION ---
        # Attraction (Force pulling to the consensus goal)
        f_att = 0.5 * (self.belief_goal - self.pos)
        
        # Repulsion (Force pushing away from obstacles)
        f_rep = np.zeros(2)
        for obs in obstacles:
            dist = np.linalg.norm(self.pos - obs)
            if dist < self.influence_dist:
                # Force is inverse square: stronger when closer
                f_rep += 10.0 * (1.0/dist - 1.0/self.influence_dist) * (self.pos - obs) / dist**3

        # --- BOIDS SOCIAL LOGIC ---
        f_sep, f_ali, f_coh = np.zeros(2), np.zeros(2), np.zeros(2)
        if neighbors:
            avg_pos = np.mean([n.pos for n in neighbors], axis=0)
            avg_vel = np.mean([n.vel for n in neighbors], axis=0)
            
            f_ali = 0.1 * (avg_vel - self.vel) # Match speed
            f_coh = 0.05 * (avg_pos - self.pos) # Move to center
            
            for n in neighbors:
                dist = np.linalg.norm(self.pos - n.pos)
                if dist < self.safe_dist:
                    f_sep += (self.pos - n.pos) / (dist**2) # Avoid crowding

        # --- CONSENSUS FILTER ---
        # Update goal belief by averaging with neighbors
        if neighbors:
            neighbor_beliefs = [n.belief_goal for n in neighbors]
            self.belief_goal = 0.8 * self.belief_goal + 0.2 * np.mean(neighbor_beliefs, axis=0)

        # Total Vector Summation
        total_force = f_att + f_rep + f_sep + f_ali + f_coh
        return total_force

    def update(self, force):
        self.vel += force
        # Speed Limiter
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed
        self.pos += self.vel

# --- SIMULATION LOOP ---
robots = [SwarmRobot(i, np.random.rand(2)*20) for i in range(10)]
obstacles = [np.array([50, 50]), np.array([30, 70])]

for step in range(100):
    for r in robots:
        # Find local neighbors within 20 units
        neighbors = [other for other in robots if other.id != r.id and np.linalg.norm(r.pos - other.pos) < 20]
        
        force = r.compute_forces(None, obstacles, neighbors)
        r.update(force)
    
    # Log the first robot's progress
    print(f"Step {step}: Robot 0 at {robots[0].pos.round(2)}, Goal Consensus: {robots[0].belief_goal.round(2)}")
