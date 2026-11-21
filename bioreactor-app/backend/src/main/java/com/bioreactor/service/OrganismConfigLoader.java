package com.bioreactor.service;

import com.bioreactor.engine.ConfigurableModel;
import com.bioreactor.engine.ConfigurableModel.*;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * Loads OrganismConfig from the seed database using direct SQL queries.
 * This avoids the JPA entity boilerplate for the V2 seed tables while
 * still making the 9 organisms available to the simulation engine.
 */
@Service
public class OrganismConfigLoader {

    private final JdbcTemplate jdbc;

    public OrganismConfigLoader(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    /** Load a single organism by ID. */
    public OrganismConfig load(String organismId) {
        if (organismId == null) throw new IllegalArgumentException("organismId must not be null");
        // ── organism row ──
        Map<String, Object> org = jdbc.queryForMap(
            "SELECT id, name, organism_type, gram_stain, description, genome_model FROM organisms WHERE id = ?",
            organismId);

        // ── growth params ──
        Map<String, Object> gp = queryOptional(
            "SELECT * FROM organism_growth_params WHERE organism_id = ?", organismId);
        String growthModel = s(gp, "model", "monod");
        double muMax = d(gp, "mu_max", 0.5);
        double kS    = d(gp, "K_s", 0.05);
        Double kI    = dOrNull(gp, "K_i");
        double yXS   = d(gp, "Y_XS", 0.45);
        double mS    = d(gp, "m_s", 0.02);
        Double xMax  = dOrNull(gp, "X_max");
        Double n     = dOrNull(gp, "n");

        // ── death params ──
        Map<String, Object> dp = queryOptional(
            "SELECT * FROM organism_death_params WHERE organism_id = ?", organismId);
        double kdMax  = d(dp, "k_d_max", 0.01);
        double kLysis = d(dp, "k_lysis", 0.005);

        // ── substrates ──
        List<SubstrateConfig> substrates = new ArrayList<>();
        jdbc.query(
            "SELECT substrate_id, q_S_max, K_m, priority, Y_XS FROM organism_substrates WHERE organism_id = ? ORDER BY priority",
            rs -> {
                substrates.add(new SubstrateConfig(
                    rs.getString("substrate_id"), rs.getString("substrate_id"),
                    rs.getDouble("q_S_max"), rs.getDouble("K_m"),
                    rs.getInt("priority")));
            }, organismId);

        // ── products ──
        List<ProductConfig> products = new ArrayList<>();
        List<ProductConfig> byproducts = new ArrayList<>();
        jdbc.query(
            "SELECT product_name, product_type, model, alpha, beta, Y_PS, q_S_crit, inhibition_type, K_p FROM organism_products WHERE organism_id = ?",
            rs -> {
                String productType = rs.getString("product_type");
                var pc = new ProductConfig(
                    rs.getString("product_name"), productType,
                    rs.getDouble("alpha"), rs.getDouble("beta"),
                    rs.getDouble("Y_PS"),
                    (Double) rs.getObject("q_S_crit"),
                    rs.getString("inhibition_type"),
                    (Double) rs.getObject("K_p"));
                if ("product".equals(productType)) products.add(pc);
                else byproducts.add(pc);
            }, organismId);

        // ── environment ──
        Map<String, Object> env = queryOptional(
            "SELECT * FROM organism_environment WHERE organism_id = ?", organismId);
        double tMin = d(env, "T_min", 15.0);
        double tOpt = d(env, "T_opt", 37.0);
        double tMax = d(env, "T_max", 45.0);
        double pHMin = d(env, "pH_min", 5.0);
        double pHOpt = d(env, "pH_opt", 7.0);
        double pHMax = d(env, "pH_max", 9.0);
        double pHSigma = d(env, "pH_sigma", 0.8);

        // ── oxygen ──
        Map<String, Object> o2 = queryOptional(
            "SELECT * FROM organism_oxygen_params WHERE organism_id = ?", organismId);
        Double qO2Max = dOrNull(o2, "q_O2_max");
        Double yXO    = dOrNull(o2, "Y_XO");
        Double mO     = dOrNull(o2, "m_O");

        // ── heat ──
        Map<String, Object> heat = queryOptional(
            "SELECT * FROM organism_heat_params WHERE organism_id = ?", organismId);
        Double yQX = dOrNull(heat, "Y_QX");
        Double yQS = dOrNull(heat, "Y_QS");

        return new OrganismConfig(
            (String) org.get("id"), (String) org.get("name"),
            growthModel, muMax, kS, kI, yXS, mS, xMax, n,
            kdMax, kLysis,
            substrates, products, byproducts,
            tMin, tOpt, tMax, pHMin, pHOpt, pHMax, pHSigma,
            qO2Max, yXO, mO, yQX, yQS);
    }

    /** Load all organism IDs. */
    public List<String> listOrganisms() {
        return jdbc.queryForList("SELECT id FROM organisms ORDER BY id", String.class);
    }

    // ── helpers ──
    private Map<String, Object> queryOptional(String sql, Object... args) {
        List<Map<String, Object>> rows = jdbc.queryForList(sql, args);
        return rows.isEmpty() ? Map.of() : rows.get(0);
    }

    private double d(Map<String, Object> row, String col, double def) {
        Object v = row.get(col);
        if (v instanceof Number n) return n.doubleValue();
        return def;
    }

    private Double dOrNull(Map<String, Object> row, String col) {
        Object v = row.get(col);
        if (v instanceof Number n) return n.doubleValue();
        return null;
    }

    private String s(Map<String, Object> row, String col, String def) {
        Object v = row.get(col);
        return v != null ? v.toString() : def;
    }
}
