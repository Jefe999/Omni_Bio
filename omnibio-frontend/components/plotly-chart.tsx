"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Download, Loader2 } from "lucide-react"

// Dynamic import for Plotly to avoid SSR issues
const loadPlotly = async () => {
  try {
    const Plotly = await import("plotly.js-dist-min")
    return Plotly.default || Plotly
  } catch (error) {
    console.error("Failed to load Plotly:", error)
    throw error
  }
}

interface PlotlyChartProps {
  data: any[]
  layout: any
  config?: any
}

export function PlotlyChart({ data, layout, config = {} }: PlotlyChartProps) {
  const plotRef = useRef<HTMLDivElement>(null)
  const [plotly, setPlotly] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadPlotly()
      .then((Plotly) => {
        setPlotly(Plotly)
        setLoading(false)
      })
      .catch((err) => {
        console.error("Error loading Plotly:", err)
        setError("Failed to load chart library")
        setLoading(false)
      })
  }, [])

  useEffect(() => {
    if (plotly && plotRef.current && !loading) {
      try {
        const mergedLayout = {
          ...layout,
          autosize: true,
          margin: { l: 60, r: 50, b: 80, t: 140, pad: 4 },
          font: { family: "Inter, sans-serif", size: 12 },
          title: {
            ...layout.title,
            font: { family: "Inter, sans-serif", size: 16, color: "#1F2937" },
            x: 0.5,
            xanchor: "center",
          },
        }

        const mergedConfig = {
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: [
            "sendDataToCloud",
            "toggleSpikelines",
          ],
          ...config,
        }

        plotly.newPlot(plotRef.current, data, mergedLayout, mergedConfig)

        const resizeHandler = () => {
          if (plotRef.current && plotly) {
            plotly.Plots.resize(plotRef.current)
          }
        }

        window.addEventListener("resize", resizeHandler)

        return () => {
          window.removeEventListener("resize", resizeHandler)
          if (plotRef.current && plotly) {
            plotly.purge(plotRef.current)
          }
        }
      } catch (err) {
        console.error("Error creating plot:", err)
        setError("Failed to create chart")
      }
    }
  }, [data, layout, config, plotly, loading])

  const downloadAsPNG = () => {
    if (plotRef.current && plotly) {
      try {
        let filename = "plot"

        if (layout.title) {
          if (typeof layout.title === "string") {
            filename = layout.title.replace(/<[^>]*>/g, "")
          } else if (typeof layout.title === "object" && layout.title.text) {
            filename = layout.title.text.replace(/<[^>]*>/g, "")
          }
        }

        plotly.downloadImage(plotRef.current, {
          format: "png",
          width: 1200,
          height: 800,
          filename,
        })
      } catch (err) {
        console.error("Error downloading chart:", err)
      }
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[400px] w-full">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading chart...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-[400px] w-full border border-gray-200 rounded">
        <div className="text-center">
          <p className="text-red-500 mb-2">Chart Error</p>
          <p className="text-sm text-gray-600">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative">
      <div className="absolute top-2 right-2 z-10">
        <Button variant="ghost" size="sm" onClick={downloadAsPNG}>
          <Download className="h-4 w-4" />
          <span className="sr-only">Download as PNG</span>
        </Button>
      </div>
      <div ref={plotRef} className="w-full h-[400px]" />
    </div>
  )
}
