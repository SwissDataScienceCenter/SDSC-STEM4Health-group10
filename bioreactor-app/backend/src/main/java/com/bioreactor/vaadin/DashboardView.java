package com.bioreactor.vaadin;

import com.bioreactor.service.ProcessRunner;
import com.bioreactor.service.ProcessRunner.*;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.combobox.ComboBox;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.html.*;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.tabs.Tab;
import com.vaadin.flow.component.tabs.Tabs;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.router.Route;

import java.util.List;
import java.util.Map;

/**
 * Manufacturing process monitoring dashboard.
 * Route: http://localhost:8080/
 */
@Route("")
public class DashboardView extends VerticalLayout {

    private final ProcessRunner runner;

    private final ComboBox<String> runSelector = new ComboBox<>("Batch Run");
    private final ComboBox<String> organismSelector = new ComboBox<>("Organism");
    private final TextField tEndField = new TextField("Horizon (days)", "20", "20");

    private final Grid<FindingInfo> findingGrid = new Grid<>(FindingInfo.class, false);
    private final Grid<Deviation> deviationGrid = new Grid<>(Deviation.class, false);
    private final Grid<Metric> metricGrid = new Grid<>(Metric.class, false);
    private final Grid<PredictionPoint> predictionGrid = new Grid<>(PredictionPoint.class, false);
    private final Grid<TrendReport> trendGrid = new Grid<>(TrendReport.class, false);

    private Span statusText = new Span("Ready");

    public DashboardView(ProcessRunner runner) {
        this.runner = runner;
        setSizeFull();
        setPadding(true);
        setSpacing(true);
        getStyle().set("background", "#f5f7fa");

        H1 header = new H1("Bioreactor Manufacturing Monitor");
        header.getStyle().set("color", "#1a3a5c");

        List<Map<String, Object>> runs = runner.listRuns();
        runSelector.setItems(runs.stream().map(r -> {
            Object id = r.get("id");
            return id != null ? id.toString() : "";
        }).filter(s -> !s.isEmpty()).toList());

        organismSelector.setItems("ecoli_k12", "cho_cells", "yeast_s288c");
        organismSelector.setValue("ecoli_k12");

        Tab monitorTab = new Tab("Monitoring");
        Tab trendTab = new Tab("Trending");
        Tab predictTab = new Tab("Prediction");
        Tabs tabs = new Tabs(monitorTab, trendTab, predictTab);
        tabs.getStyle().set("border-bottom", "2px solid #1a3a5c");

        VerticalLayout monitorContent = buildMonitoringTab();
        VerticalLayout trendContent = buildTrendingTab();
        VerticalLayout predictContent = buildPredictionTab();

        Div contentArea = new Div(monitorContent);
        contentArea.setSizeFull();
        tabs.addSelectedChangeListener(ev -> {
            contentArea.removeAll();
            Tab s = ev.getSelectedTab();
            if (s == monitorTab) contentArea.add(monitorContent);
            else if (s == trendTab) contentArea.add(trendContent);
            else contentArea.add(predictContent);
        });

        add(header, statusText, tabs, contentArea);
        setFlexGrow(1, contentArea);
    }

    private VerticalLayout buildMonitoringTab() {
        var l = new VerticalLayout(); l.setSizeFull();
        var c = new HorizontalLayout();
        var btn = new Button("Run Analysis", ev -> runMonitoringAnalysis());
        btn.getStyle().set("background", "#1a3a5c"); btn.getStyle().set("color", "white");
        c.add(runSelector, organismSelector, btn); c.setAlignItems(Alignment.BASELINE);

        findingGrid.addColumn(FindingInfo::elapsedTime).setHeader("Time (d)");
        findingGrid.addColumn(FindingInfo::parameterCode).setHeader("Param");
        findingGrid.addColumn(FindingInfo::severity).setHeader("Severity");
        findingGrid.addColumn(FindingInfo::kind).setHeader("Type");
        findingGrid.addColumn(FindingInfo::message).setHeader("Message").setAutoWidth(true);

        deviationGrid.addColumn(Deviation::time).setHeader("Time");
        deviationGrid.addColumn(Deviation::parameter).setHeader("Param");
        deviationGrid.addColumn(d -> fmt(d.measured())).setHeader("Measured");
        deviationGrid.addColumn(d -> fmt(d.predicted())).setHeader("Predicted");
        deviationGrid.addColumn(d -> String.format("%.1f%%", d.relErrorPct())).setHeader("Error");
        deviationGrid.addColumn(Deviation::severity).setHeader("Severity");

        metricGrid.addColumn(Metric::code).setHeader("Metric");
        metricGrid.addColumn(m -> fmt(m.value())).setHeader("Value");
        metricGrid.addColumn(Metric::unit).setHeader("Unit");

        l.add(c, new H3("Findings"), findingGrid,
            new H3("Deviations"), deviationGrid, new H3("Metrics"), metricGrid);
        return l;
    }

    private VerticalLayout buildTrendingTab() {
        var l = new VerticalLayout(); l.setSizeFull();
        var c = new HorizontalLayout();
        var pf = new TextField("Process ID", "", ""); pf.setWidth("350px");
        var pb = new ComboBox<>("Parameter", "peak_vcd", "final_vcd", "vcd", "substrate");
        pb.setValue("peak_vcd");
        var btn = new Button("Run Trending", ev -> runTrending(pf.getValue(), pb.getValue()));
        btn.getStyle().set("background", "#1a3a5c"); btn.getStyle().set("color", "white");
        c.add(pf, pb, btn); c.setAlignItems(Alignment.BASELINE);

        trendGrid.addColumn(TrendReport::parameterCode).setHeader("Parameter");
        trendGrid.addColumn(t -> fmt(t.mean())).setHeader("Mean");
        trendGrid.addColumn(t -> fmt(t.sigma())).setHeader("Sigma");
        trendGrid.addColumn(t -> fmt(t.ucl())).setHeader("UCL");
        trendGrid.addColumn(t -> fmt(t.lcl())).setHeader("LCL");
        trendGrid.addColumn(t -> String.format("%.2f", t.cpk())).setHeader("Cpk");

        l.add(c, new H3("Trend Summary"), trendGrid);
        return l;
    }

    private VerticalLayout buildPredictionTab() {
        var l = new VerticalLayout(); l.setSizeFull();
        var c = new HorizontalLayout();
        var btn = new Button("Run Prediction", ev -> runPrediction());
        btn.getStyle().set("background", "#1a3a5c"); btn.getStyle().set("color", "white");
        c.add(runSelector, organismSelector, tEndField, btn); c.setAlignItems(Alignment.BASELINE);

        predictionGrid.addColumn(PredictionPoint::time).setHeader("Time");
        predictionGrid.addColumn(PredictionPoint::parameterCode).setHeader("Param");
        predictionGrid.addColumn(p -> fmt(p.value())).setHeader("Value");
        predictionGrid.addColumn(p -> fmt(p.lower())).setHeader("Lower");
        predictionGrid.addColumn(p -> fmt(p.upper())).setHeader("Upper");

        l.add(c, new H3("Predictions"), predictionGrid);
        return l;
    }

    private void runMonitoringAnalysis() {
        try {
            long t0 = System.currentTimeMillis();
            var r = runner.analyzeRun(runSelector.getValue(), organismSelector.getValue());
            long ms = System.currentTimeMillis() - t0;
            findingGrid.setItems(r.findings());
            deviationGrid.setItems(r.deviations());
            metricGrid.setItems(r.metrics());
            statusText.setText(String.format("Done in %d ms | %d findings | %d devs | %.1f days",
                ms, r.findings().size(), r.deviations().size(), r.tEnd()));
            statusText.getStyle().set("color", "green");
        } catch (Exception e) {
            statusText.setText("Error: " + e.getMessage());
            statusText.getStyle().set("color", "red");
        }
    }

    private void runTrending(String pid, String param) {
        try {
            var r = runner.trendAnalysis(pid, param);
            trendGrid.setItems(List.of(r));
            statusText.setText(String.format("mean=%.2f sigma=%.2f Cpk=%.2f", r.mean(), r.sigma(), r.cpk()));
            statusText.getStyle().set("color", "green");
        } catch (Exception e) {
            statusText.setText("Error: " + e.getMessage());
            statusText.getStyle().set("color", "red");
        }
    }

    private void runPrediction() {
        try {
            double te = Double.parseDouble(tEndField.getValue());
            var r = runner.predictRun(runSelector.getValue(), organismSelector.getValue(), List.of(), te);
            predictionGrid.setItems(r.predictions());
            statusText.setText("Predicted " + r.predictions().size() + " points");
            statusText.getStyle().set("color", "green");
        } catch (Exception e) {
            statusText.setText("Error: " + e.getMessage());
            statusText.getStyle().set("color", "red");
        }
    }

    private static String fmt(double v) { return String.format("%.4f", v); }
}
