package com.bioreactor.repository;

import com.bioreactor.model.Organism;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface OrganismRepository extends JpaRepository<Organism, String> {
    List<Organism> findByOrganismType(String type);
}
