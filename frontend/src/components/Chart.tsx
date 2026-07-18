import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { BarChart, CandlestickChart, LineChart } from "echarts/charts";
import { GridComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsCoreOption } from "echarts/core";

echarts.use([BarChart, CandlestickChart, LineChart, GridComponent, TooltipComponent, CanvasRenderer]);

interface ChartProps { option: unknown; ariaLabel: string; className?: string }

export function Chart({ option, ariaLabel, className = "chart" }: ChartProps) {
  const element = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!element.current) return;
    const chart = echarts.init(element.current, undefined, { renderer: "canvas" });
    chart.setOption(option as EChartsCoreOption);
    const resize = () => chart.resize();
    window.addEventListener("resize", resize);
    return () => { window.removeEventListener("resize", resize); chart.dispose(); };
  }, [option]);
  return <div ref={element} className={className} role="img" aria-label={ariaLabel} />;
}
