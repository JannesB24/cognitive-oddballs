from pyhgf.model.network import Network

class RobustGHGF(Network):
    """
    A 3-level Robust gHGF designed for the Helicopter Task.
    Distinguishes between 'Volatility' (Change-points) and 'Noise' (Oddballs).
    """
    def __init__(self, initial_mean=250.0):
        # Initialize the base Network
        super().__init__(update_type="eHGF")
        
        # Node 0: Volatility Parent (x_vol)
        self.add_nodes(
            kind="continuous-state",
            node_parameters={
                "mean": -6.0,         
                "precision": 1.0, 
                "tonic_volatility": -3.0
            }
        )

        # Node 1: Noise Parent (x_noise)
        self.add_nodes(
            kind="continuous-state",
            node_parameters={
                "mean": -6.0,        
                "precision": 2.0,      
                "tonic_volatility": -1.5
            }
        )

        # Node 2: Mean Tracker (x_mean)
        self.add_nodes(
            kind="continuous-state",
            volatility_parents=0, 
            node_parameters={
                "mean": initial_mean, 
                "precision": 0.999     
            }
        )

        # Node 3: Observation Node (u_bag)
        self.add_nodes(
            kind="continuous-state",
            value_parents=2,      
            volatility_parents=1  
        )

        # Compile JAX functions
        self.create_belief_propagation_fn()

def build_robust_ghgf(initial_mean=250.0):
    return RobustGHGF(initial_mean=initial_mean)