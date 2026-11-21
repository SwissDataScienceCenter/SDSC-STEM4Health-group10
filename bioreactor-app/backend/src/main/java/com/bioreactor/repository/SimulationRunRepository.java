package com.bioreactor.repository;

import com.bioreactor.model.SimulationRun;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface SimulationRunRepository extends JpaRepository<SimulationRun, Integer> {
    List<SimulationRun> findByStatus(String status);
    List<SimulationRun> findByConfigIdOrderByCreatedAtDesc(Integer configId);
}
