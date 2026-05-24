import math
from typing import List, Dict, Tuple

class ViterbiSolver:
    def __init__(self, states: List[str], transitions: Dict[str, Dict[str, float]], emissions: Dict[str, Dict[str, float]], initial_probs: Dict[str, float]):
        """
        Initializes the HMM parameters and converts them directly to log10 space.
        """
        self.states = states
        
        # Convert initial probabilities to log10
        self.log_initial = {}
        for s in states:
            val = initial_probs.get(s, 0.0)
            self.log_initial[s] = math.log10(val) if val > 0 else -float('inf')
            
        # Convert transitions to log10: transitions[from_state][to_state]
        self.log_trans = {s: {} for s in states}
        for s_from in states:
            for s_to in states:
                val = transitions[s_from].get(s_to, 0.0)
                self.log_trans[s_from][s_to] = math.log10(val) if val > 0 else -float('inf')
                
        # Convert emissions to log10: emissions[state][observation]
        self.log_emiss = {s: {} for s in states}
        for s in states:
            for obs, val in emissions[s].items():
                self.log_emiss[s][obs] = math.log10(val) if val > 0 else -float('inf')

    def solve(self, sequence: List[str], verbose: bool = True) -> Tuple[List[str], float]:
        """
        Runs the Viterbi algorithm over an observed sequence.
        Returns the optimal state sequence and its raw decimal probability.
        """
        n = len(sequence)
        if n == 0:
            return [], 1.0

        # Viterbi trellis tables stored as lists of dictionaries
        # trellis[t][state] = log10_probability
        trellis = [{} for _ in range(n + 1)]
        backpointers = [{} for _ in range(n + 1)]

        # --- Step 0: Initialization ---
        for s in self.states:
            trellis[0][s] = self.log_initial[s]
            
        if verbose:
            print("=" * 90)
            print(" INITIALIZATION (t=0) ")
            print("=" * 90)
            for s in self.states:
                print(f"  State {s}: log10_prob = {trellis[0][s]}")
            print()

        # --- Forward Pass ---
        for t in range(1, n + 1):
            obs = sequence[t - 1]
            if verbose:
                print("=" * 90)
                print(f" TIME STEP t={t} | Observing item: '{obs}'")
                print("=" * 90)

            for s_curr in self.states:
                if verbose:
                    print(f"  --> Calculating best path to land in State [{s_curr}]:")
                
                best_path_val = -float('inf')
                best_prev_state = None
                
                # Check incoming paths from all possible previous states
                for s_prev in self.states:
                    prev_val = trellis[t - 1][s_prev]
                    trans_val = self.log_trans[s_prev][s_curr]
                    total_incoming = prev_val + trans_val
                    
                    if verbose:
                        print(f"      From State [{s_prev}]: {prev_val:.4f} (prev) + {trans_val:.4f} (trans) = {total_incoming:.4f}")
                        
                    if total_incoming > best_path_val:
                        best_path_val = total_incoming
                        best_prev_state = s_prev
                
                # Multiply (add in log-space) the emission probability
                emiss_val = self.log_emiss[s_curr].get(obs, -float('inf'))
                final_cell_val = best_path_val + emiss_val
                
                trellis[t][s_curr] = final_cell_val
                backpointers[t][s_curr] = best_prev_state
                
                if verbose:
                    print(f"      MAX incoming is {best_path_val:.4f} from State [{best_prev_state}]")
                    print(f"      Add Emission log10_e({obs}|{s_curr}) = {emiss_val:.4f}")
                    print(f"      [RESULT] trellis[t={t}][{s_curr}] = {final_cell_val:.4f}\n")

        # --- Traceback Selection ---
        if verbose:
            print("=" * 90)
            print(" TRACEBACK SELECTION ")
            print("=" * 90)

        best_final_val = -float('inf')
        best_final_state = None
        
        for s in self.states:
            if trellis[n][s] > best_final_val:
                best_final_val = trellis[n][s]
                best_final_state = s
                
        if verbose:
            for s in self.states:
                print(f"  Final cell value at [{s}]: {trellis[n][s]:.4f}")
            print(f"  >> Winning Final State is [{best_final_state}] with log10 score: {best_final_val:.4f}\n")

        # --- Traceback Execution ---
        path = []
        curr_state = best_final_state
        
        for t in range(n, 0, -1):
            path.append(curr_state)
            prev_state = backpointers[t][curr_state]
            if verbose and t > 1:
                print(f"  At t={t}, state is [{curr_state}]. Pointer points back to [{prev_state}]")
            curr_state = prev_state
            
        path.reverse()
        raw_prob = 10 ** best_final_val

        if verbose:
            print("\n" + "=" * 90)
            print(f" FINAL OUTPUT PATH: {' -> '.join(path)}")
            print(f" Raw decimal probability: {raw_prob:.4e}")
            print("=" * 90 + "\n")

        return path, raw_prob
    

if __name__ == "__main__":
    # Define states
    states = ['H', 'L']
    
    # Define sequence
    observed_sequence = list("ACCGTGCA")
    
    # Define parameters exactly from the assignment question
    initial_probabilities = {'H': 1.0, 'L': 0.0}
    
    transition_matrix = {
        'H': {'H': 0.5, 'L': 0.5},
        'L': {'H': 0.4, 'L': 0.6}
    }
    
    emission_matrix = {
        'H': {'A': 0.2, 'C': 0.3, 'G': 0.3, 'T': 0.2},
        'L': {'A': 0.3, 'C': 0.2, 'G': 0.2, 'T': 0.3}
    }
    
    # Run Solver
    solver = ViterbiSolver(states, transition_matrix, emission_matrix, initial_probabilities)
    
    # Set verbose=True to view the entire logging breakout step-by-step
    optimal_path, final_decimal_prob = solver.solve(observed_sequence, verbose=True)   