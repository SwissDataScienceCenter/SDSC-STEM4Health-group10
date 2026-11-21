package com.bioreactor.model;

import jakarta.persistence.*;

@Entity
@Table(name = "organism_growth_params")
public class OrganismGrowthParams {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;
    
    @Column(name = "organism_id")
    private String organismId;
    
    private String model;
    
    @Column(name = "mu_max")
    private Double muMax;
    
    @Column(name = "k_s")
    private Double kS;
    
    @Column(name = "k_i")
    private Double kI;
    
    @Column(name = "y_xs")
    private Double yXS;
    
    @Column(name = "m_s")
    private Double mS;
    
    @Column(name = "x_max")
    private Double xMax;
    
    private Double n;

    public OrganismGrowthParams() {}

    public Integer getId() { return id; }
    public String getOrganismId() { return organismId; }
    public String getModel() { return model; }
    public Double getMuMax() { return muMax; }
    public Double getkS() { return kS; }
    public Double getkI() { return kI; }
    public Double getyXS() { return yXS; }
    public Double getmS() { return mS; }
    public Double getxMax() { return xMax; }
    public Double getN() { return n; }
}
