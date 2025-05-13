# Advanced Agent Features

JaxABM now supports advanced agent features that go beyond the standard setup and step methods. Agents can now have custom methods that can be called outside of the step function during simulation.

## Custom Agent Methods

Agents can have custom methods that define behavior beyond what happens during the regular step function:

```python
class MyAgent(jx.Agent):
    def setup(self):
        """Initialize agent state."""
        return {
            'x': 0.5,
            'y': 0.5,
            'energy': 100.0
        }
    
    def step(self, model_state):
        """Standard update during each step."""
        return {
            'x': self._state['x'] + 0.01,
            'y': self._state['y'] + 0.01,
            'energy': self._state['energy'] - 1.0
        }
    
    def boost_energy(self, amount=10.0):
        """Custom method to boost agent energy.
        
        This can be called from outside the step function.
        """
        current_energy = self._state['energy']
        new_energy = current_energy + amount
        
        # Update state using update_state method
        self.update_state({'energy': new_energy})
        
        return new_energy
```

## Accessing and Using Agent Instances

You can access individual agent instances in your model and call their custom methods:

```python
class MyModel(jx.Model):
    def setup(self):
        """Set up model with agents."""
        self.agents = self.add_agents(10, MyAgent)
    
    def step(self):
        """Execute model logic each step."""
        # Every 5 steps, boost a random agent's energy
        if self.env.time % 5 == 0:
            # Choose a random agent
            agent_id = random.randint(0, len(self.agents) - 1)
            
            # Get the agent instance
            agent = self.get_agent('agents', agent_id)
            
            # Call custom method
            if agent:
                new_energy = agent.boost_energy(20.0)
                print(f"Boosted agent {agent_id}'s energy to {new_energy}")
```

## Iterating Over AgentList

You can also iterate over an `AgentList` to access all agent instances:

```python
# Iterate over agents and call a custom method on each
for agent in model.agents:
    if agent.energy < 50:
        agent.boost_energy(10.0)
```

## State Updates via Custom Methods

When you update an agent's state through a custom method, the changes are propagated to the underlying model state. There are two ways to update state:

1. Using direct attribute assignment (recommended):
```python
agent.energy = 100.0  # This updates the state internally
```

2. Using the update_state method:
```python
agent.update_state({'energy': 100.0, 'x': 0.5})
```

## Limitations

While custom methods provide flexibility, keep in mind:

1. JAX constraints still apply to the state values (must be JAX-compatible types)
2. Custom method calls are not part of the JIT-compiled execution path, so they may be less performant
3. State updates through custom methods are more flexible but less efficient than vectorized updates in the step function

For performance-critical code, it's best to implement behavior in the step function where possible.

## Complete Example

See `examples/custom_agent_methods.py` for a complete example of using custom agent methods.