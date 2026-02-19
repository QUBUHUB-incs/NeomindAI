class SwarmRobot:
    def __init__(self, robot_id, start_pos):
        self.id = robot_id
        self.pos = np.array(start_pos, dtype=float)
        self.vel = np.zeros(2)
        # Initially, robots are in "Roaming Mode" with a default center-map belief
        self.belief_goal = np.array([50.0, 50.0]) 
        self.found_target = False 
        self.pid = PID(kp=0.6, ki=0.01, kd=0.1)

    def sense_vision(self, real_target_pos):
        """Simulates a camera/vision sensor detecting a target in FOV."""
        dist_to_target = np.linalg.norm(self.pos - real_target_pos)
        # If target is within 15 units, the vision sensor 'locks' on
        if dist_to_target < 15.0:
            self.belief_goal = real_target_pos
            self.found_target = True
            return True
        return False

    def compute_forces(self, obstacles, neighbors):
        # 1. Consensus Logic: Listen to neighbors
        # If a neighbor has found a target, their belief is weighted more heavily
        if neighbors:
            # Filter for neighbors who have 'found_target' = True
            scouts = [n for n in neighbors if getattr(n, 'found_target', False)]
            if scouts:
                # Rapidly pull consensus toward the scout's discovery
                target_avg = np.mean([s.belief_goal for s in scouts], axis=0)
                self.belief_goal = 0.7 * self.belief_goal + 0.3 * target_avg
                self.found_target = True # Spread the 'alert'
            else:
                # Standard slow consensus for general alignment
                avg_belief = np.mean([n.belief_goal for n in neighbors], axis=0)
                self.belief_goal = 0.98 * self.belief_goal + 0.02 * avg_belief

        # 2. Navigation Forces (APF + Boids)
        f_att = 0.5 * (self.belief_goal - self.pos)
        f_rep = np.zeros(2)
        for obs in obstacles:
            dist = np.linalg.norm(self.pos - obs)
            if dist < 12.0:
                f_rep += 20.0 * (1.0/dist - 1.0/12.0) * (self.pos - obs) / dist**3
        
        return f_att + f_rep

    def step(self, real_target, obstacles, neighbors):
        self.sense_vision(real_target) # Vision Layer
        force = self.compute_forces(obstacles, neighbors)
        accel = self.pid.compute(force, self.vel)
        self.vel += accel
        self.pos += self.vel
