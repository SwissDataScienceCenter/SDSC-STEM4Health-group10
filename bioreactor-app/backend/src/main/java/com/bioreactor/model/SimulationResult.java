package com.bioreactor.model;

import jakarta.persistence.*;

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
    
    @Column(name = "ph")
    private Double pH;
    
    private Double mu;
    
    @Column(name = "q_s")
    private Double qS;
    
    @Column(name = "q_p")
    private Double qP;
    
    @Column(name = "q_b")
    private Double qB;
    
    @Column(name = "q_o2")
    private Double qO2;
    
    private Double viability;

    public SimulationResult() {}

    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    
    public Integer getSimulationId() { return simulationId; }
    public void setSimulationId(Integer v) { this.simulationId = v; }
    
    public Double getTime() { return time; }
    public void setTime(Double v) { this.time = v; }
    
    public Double getxV() { return xV; }
    public void setxV(Double v) { this.xV = v; }
    
    public Double getxD() { return xD; }
    public void setxD(Double v) { this.xD = v; }
    
    public Double getSubstrate() { return substrate; }
    public void setSubstrate(Double v) { this.substrate = v; }
    
    public Double getProduct() { return product; }
    public void setProduct(Double v) { this.product = v; }
    
    public Double getByproduct() { return byproduct; }
    public void setByproduct(Double v) { this.byproduct = v; }
    
    public Double getDissolvedO2() { return dissolvedO2; }
    public void setDissolvedO2(Double v) { this.dissolvedO2 = v; }
    
    public Double getVolume() { return volume; }
    public void setVolume(Double v) { this.volume = v; }
    
    public Double getTemperature() { return temperature; }
    public void setTemperature(Double v) { this.temperature = v; }
    
    public Double getpH() { return pH; }
    public void setpH(Double v) { this.pH = v; }
    
    public Double getMu() { return mu; }
    public void setMu(Double v) { this.mu = v; }
    
    public Double getqS() { return qS; }
    public void setqS(Double v) { this.qS = v; }
    
    public Double getqP() { return qP; }
    public void setqP(Double v) { this.qP = v; }
    
    public Double getqB() { return qB; }
    public void setqB(Double v) { this.qB = v; }
    
    public Double getqO2() { return qO2; }
    public void setqO2(Double v) { this.qO2 = v; }
    
    public Double getViability() { return viability; }
    public void setViability(Double v) { this.viability = v; }
}
