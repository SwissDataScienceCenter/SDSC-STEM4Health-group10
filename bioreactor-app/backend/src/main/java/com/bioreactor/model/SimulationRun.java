package com.bioreactor.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "simulation_runs")
public class SimulationRun {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;
    
    @Column(name = "config_id")
    private Integer configId;
    
    private String name;
    private String description;
    
    @Column(name = "t_start")
    private Double tStart = 0.0;
    
    @Column(name = "t_end")
    private Double tEnd;
    
    private Double dt = 0.1;
    private String solver = "LSODA";
    private String status = "pending";
    
    @Column(name = "created_at")
    private LocalDateTime createdAt = LocalDateTime.now();
    
    @Column(name = "started_at")
    private LocalDateTime startedAt;
    
    @Column(name = "completed_at")
    private LocalDateTime completedAt;

    public SimulationRun() {}

    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    public Integer getConfigId() { return configId; }
    public void setConfigId(Integer configId) { this.configId = configId; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public Double gettStart() { return tStart; }
    public void settStart(Double tStart) { this.tStart = tStart; }
    public Double gettEnd() { return tEnd; }
    public void settEnd(Double tEnd) { this.tEnd = tEnd; }
    public Double getDt() { return dt; }
    public void setDt(Double dt) { this.dt = dt; }
    public String getSolver() { return solver; }
    public void setSolver(String solver) { this.solver = solver; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getStartedAt() { return startedAt; }
    public void setStartedAt(LocalDateTime startedAt) { this.startedAt = startedAt; }
    public LocalDateTime getCompletedAt() { return completedAt; }
    public void setCompletedAt(LocalDateTime completedAt) { this.completedAt = completedAt; }
}
