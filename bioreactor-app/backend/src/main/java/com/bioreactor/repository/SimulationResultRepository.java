package com.bioreactor.repository;

import com.bioreactor.model.SimulationResult;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface SimulationResultRepository extends JpaRepository<SimulationResult, Long> {
    List<SimulationResult> findBySimulationIdOrderByTimeAsc(Integer simulationId);
    void deleteBySimulationId(Integer simulationId);
}
