package com.bioreactor.model;

import jakarta.persistence.*;
import java.util.Map;

@Entity
@Table(name = "simulation_results")
public class SimulationResult {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(columnDefinition = "serial")
    private Integer id;
    
    @Column(name = "simulation_id")
    private Integer simulationId;
    
    @Column(name = "t")
    private Double time;
    
    @Column(name = "x_v")
    private Double xV;
    
    @Column(name = "x_d")
    private Double xD;
    
    @Column(name = "s")
    private Double substrate;
    
    @Column(name = "p")
    private Double product;
    
    @Column(name = "b")
    private Double byproduct;
    
    @Column(name = "dissolved_o2")
    private Double dissolvedO2;
    
    @Column(name = "v")
    private Double volume;
    
    private Double temperature;
    private Double pH;
    private Double mu;
    private Double qS;
    private Double qP;
    private Double qB;
    
    @Column(name = "q_o2")
    private Double qO2;
    
    private Double viability;

    public SimulationResult() {}

    public Integer getId() { return id; }
    public Integer getSimulationId() { return simulationId; }
    public Double getTime() { return time; }
    public Double getxV() { return xV; }
    public Double getxD() { return xD; }
    public Double getSubstrate() { return substrate; }
    public Double getProduct() { return product; }
    public Double getByproduct() { return byproduct; }
    public Double getDissolvedO2() { return dissolvedO2; }
    public Double getVolume() { return volume; }
    public Double getTemperature() { return temperature; }
    public Double getpH() { return pH; }
    public Double getMu() { return mu; }
    public Double getqS() { return qS; }
    public Double getqP() { return qP; }
    public Double getqB() { return qB; }
    public Double getqO2() { return qO2; }
    public Double getViability() { return viability; }
}
