# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence

import numpy as np
import time
import torch
from datetime import datetime
import random

import evaluator
import programs_database


class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    raise NotImplementedError('Must provide a language model.')

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class SimpleLLM(LLM):
  """A simple language model implementation for testing."""

  def _draw_sample(self, prompt: str) -> str:
    """Returns a test implementation with a simple strategy."""
    return """def evolve_grid(grid_description: tuple[int, int]) -> tuple[int, int]:
    size, seed = grid_description
    import random
    
    # Strategy 1: Alternating pattern with random offset
    if random.random() < 0.5:
        # Create a new seed that will produce an alternating pattern
        new_seed = random.randint(1, 10000)
        return (size, new_seed)
    
    # Strategy 2: Modify existing seed slightly to explore nearby solutions
    else:
        # Small modification to explore similar solutions
        new_seed = seed + random.randint(-100, 100)
        return (size, new_seed)"""


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = SimpleLLM(samples_per_prompt)
    self._start_time = time.time()
    self._total_attempts = 0
    self._successful_attempts = 0
    self._last_success_time = time.time()
    self._best_score = float('-inf')
    self._last_progress_update = time.time()
    self._progress_update_interval = 5  # seconds

  def _get_gpu_memory_usage(self) -> float:
    """Get current GPU memory usage in GB."""
    if torch.backends.mps.is_available():
      return torch.mps.current_allocated_memory() / (1024**3)
    elif torch.cuda.is_available():
      return torch.cuda.memory_allocated() / (1024**3)
    return 0.0

  def _print_progress(self):
    """Print current progress metrics."""
    current_time = time.time()
    elapsed_time = current_time - self._start_time
    time_since_last_success = current_time - self._last_success_time
    success_rate = (self._successful_attempts / self._total_attempts * 100) if self._total_attempts > 0 else 0
    gpu_memory = self._get_gpu_memory_usage()

    print("\n" + "="*50)
    print(f"Progress Update - {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total Attempts: {self._total_attempts}")
    print(f"Successful Attempts: {self._successful_attempts}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Best Score: {self._best_score}")
    print(f"Time Since Last Success: {time_since_last_success:.1f}s")
    print(f"Total Time Elapsed: {elapsed_time:.1f}s")
    print(f"GPU Memory Usage: {gpu_memory:.2f}GB")
    print("="*50 + "\n")

  def update_success(self, score: float):
    """Update success metrics when a valid solution is found."""
    self._successful_attempts += 1
    self._last_success_time = time.time()
    if score > self._best_score:
      self._best_score = score
      print(f"\nNew Best Score Found: {self._best_score}")
      print(f"Total Successful Attempts: {self._successful_attempts}")
      print(f"Success Rate: {(self._successful_attempts / self._total_attempts * 100):.2f}%\n")

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    print("\nStarting FunSearch for L-shape Ramsey Grid...")
    print(f"Initial GPU Memory: {self._get_gpu_memory_usage():.2f}GB\n")
    
    while True:
      prompt = self._database.get_prompt()
      samples = self._llm.draw_samples(prompt.code)
      
      # This loop can be executed in parallel on remote evaluator machines.
      for sample in samples:
        self._total_attempts += 1
        
        # Update progress periodically
        current_time = time.time()
        if current_time - self._last_progress_update >= self._progress_update_interval:
          self._print_progress()
          self._last_progress_update = current_time
        
        chosen_evaluator = np.random.choice(self._evaluators)
        # Pass self as sampler to the evaluator
        chosen_evaluator._sampler = self
        chosen_evaluator.analyse(
            sample, prompt.island_id, prompt.version_generated)
