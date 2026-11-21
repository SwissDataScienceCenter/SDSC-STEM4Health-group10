#!/usr/bin/env python3
"""
CLI script called by Java Spring Boot to run simulations.
Reads JSON from stdin, runs simulation, writes JSON results + saves to DB.
"""
import sys
import json
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation_engine import run_simulation, SimulationConfig, BioreactorConfig, InitialConditions
from db_connector import load_organism, save_simulation_result


def main():
    input_data = sys.stdin.read().strip()
    if not input_data:
        print(json.dumps({"status": "error", "message": "No input received"}))
        sys.exit(1)
    
    try:
        params = json.loads(input_data)
    except json.JSONDecodeError as e:
        print(json.dumps({"status": "error", "message": f"Invalid JSON: {e}"}))
        sys.exit(1)
    
    organism_id = params.get("organism_id", "ecoli_k12")
    simulation_id = params.get("simulation_id")
    
    # Load organism from DB
    organism = load_organism(organism_id)
    if organism is None:
        print(json.dumps({"status": "error", "message": f"Organism '{organism_id}' not found"}))
        sys.exit(1)
    
    # Build config
    br = BioreactorConfig(
        mode=params.get("mode", "batch"),
        V_initial=params.get("V", 10.0),
        kLa=params.get("kLa", 150.0),
        T_setpoint=params.get("temperature", 37.0),
        pH_setpoint=params.get("pH", 7.0)
    )
    
    ic = InitialConditions(
        X_v=params.get("X_v", 1.0),
        S=params.get("S", 5.0),
        V=params.get("V", 10.0),
        temperature=params.get("temperature", 37.0),
        pH=params.get("pH", 7.0)
    )
    
    t_end = params.get("t_end", 24.0)
    
    config = SimulationConfig(
        organism=organism,
        bioreactor=br,
        initial=ic,
        t_end=t_end,
        dt_output=0.1
    )
    
    # Run simulation
    results = run_simulation(config)
    
    # Save to DB if simulation_id provided
    if simulation_id and results["status"] == "success":
        try:
            save_simulation_result(int(simulation_id), results)
            results["db_saved"] = True
        except Exception as e:
            results["db_saved"] = False
            results["db_error"] = str(e)
    
    # Print results
    print(json.dumps({
        "status": results["status"],
        "message": results.get("message", ""),
        "organism": organism.name,
        "final_X_v": round(results["X_v"][-1], 3),
        "final_S": round(results["substrate"][-1], 3),
        "final_B": round(results["byproduct"][-1], 3),
        "final_viability": round(results["viability"][-1], 3),
        "time_points": len(results["time"]),
        "simulation_id": simulation_id,
        "db_saved": results.get("db_saved", False)
    }))


if __name__ == "__main__":
    main()
