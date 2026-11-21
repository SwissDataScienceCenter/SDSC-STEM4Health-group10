package com.bioreactor.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "organisms")
public class Organism {
    @Id
    private String id;
    
    private String name;
    
    @Column(name = "organism_type")
    private String organismType;
    
    @Column(name = "gram_stain")
    private String gramStain;
    
    private String description;
    
    @Column(name = "genome_model")
    private String genomeModel;
    
    @Column(name = "created_at")
    private LocalDateTime createdAt;
    
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    public Organism() {}

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getOrganismType() { return organismType; }
    public void setOrganismType(String organismType) { this.organismType = organismType; }
    public String getGramStain() { return gramStain; }
    public void setGramStain(String gramStain) { this.gramStain = gramStain; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getGenomeModel() { return genomeModel; }
    public void setGenomeModel(String genomeModel) { this.genomeModel = genomeModel; }
}
