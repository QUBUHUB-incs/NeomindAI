class RobotState(enum.Enum):
    SEARCHING = 1
    GUARDING = 2
    LOW_POWER = 3  # New: Seeking charging station
    DOCKING = 4    # New: Physical connection in progress

class SwarmRobot:
    def __init__(self, robot_id, start_pos):
        self.id = robot_id
        self.pos = np.array(start_pos, dtype=float)
        self.vel = np.zeros(2)
        self.battery = 100.0  # Max charge
        self.state = RobotState.SEARCHING
        self.home_base = np.array([0.0, 0.0]) # Charging hub location
        self.critical_level = 20.0 # Return home at 20%

    def update_battery(self, dt=0.1):
        """Simulate energy drain based on movement speed."""
        drain_rate = 0.05 + (0.1 * np.linalg.norm(self.vel))
        self.battery -= drain_rate * dt
        
        if self.battery < self.critical_level and self.state != RobotState.DOCKING:
            self.state = RobotState.LOW_POWER

    def handle_docking(self, neighbors):
        """Logic for physical connection or charging station entry."""
        dist_to_hub = np.linalg.norm(self.pos - self.home_base)
        
        if self.state == RobotState.LOW_POWER and dist_to_hub < 2.0:
            self.state = RobotState.DOCKING
            self.vel = np.zeros(2) # Stop for docking
            
        if self.state == RobotState.DOCKING:
            self.battery += 2.0 # Recharge rate
            if self.battery >= 95.0:
                self.state = RobotState.SEARCHING

    def compute_forces(self, neighbors):
        # 1. Goal Selection based on State
        if self.state == RobotState.LOW_POWER:
            target = self.home_base
        else:
            target = self.belief_goal

        # 2. Navigation Force
        f_nav = 0.6 * (target - self.pos)

        # 3. Trophallaxis (Energy Sharing)
        f_share = np.zeros(2)
        for n in neighbors:
            # If a neighbor is critical, move slightly toward them to 'donate' charge
            if n.battery < 10.0 and self.battery > 50.0:
                f_share += 0.2 * (n.pos - self.pos)
        
        return f_nav + f_share

    def step(self, neighbors):
        self.update_battery()
        self.handle_docking(neighbors)
        
        if self.state != RobotState.DOCKING:
            force = self.compute_forces(neighbors)
            self.vel += force
            self.pos += self.vel
