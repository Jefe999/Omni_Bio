"use client"

import { useState, useEffect } from "react"
import MainLayout from "@/components/main-layout"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { UploadDropzone } from "@/components/upload-dropzone"
import { FileTable } from "@/components/file-table"
import { JobCard } from "@/components/job-card"
import { NewAnalysisModal } from "@/components/new-analysis-modal"
import { Upload, Plus, RefreshCw } from "lucide-react"
import { apiClient } from "@/lib/api"
import { FileInfo } from "@/lib/api"

// Job interface to match JobCard expectations exactly
interface Job {
  id: string
  name: string
  status: "running" | "completed" | "failed"
  progress: number
  type: string
  error?: string
}

export default function DashboardPage() {
  const [files, setFiles] = useState<FileInfo[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState("jobs")

  // Load initial data
  useEffect(() => {
    loadData()
  }, [])

  // Poll for job updates every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      loadJobs()
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      await Promise.all([loadFiles(), loadJobs()])
    } catch (err) {
      setError("Failed to load data")
      console.error("Failed to load data:", err)
    } finally {
      setLoading(false)
    }
  }

  const loadFiles = async () => {
    try {
      const response = await apiClient.getFiles() as any
      const filesData = response.files || []
      setFiles(filesData.map((file: any) => ({
        id: file.file_id,
        name: file.original_filename,
        type: file.file_type,
        date: new Date(file.uploaded_at).toISOString().split("T")[0],
        size: `${(file.size_bytes / (1024 * 1024)).toFixed(1)} MB`,
        file_id: file.file_id,
        original_filename: file.original_filename,
        file_type: file.file_type,
        uploaded_at: file.uploaded_at,
        size_bytes: file.size_bytes
      })))
    } catch (err) {
      console.error("Failed to load files:", err)
    }
  }

  const loadJobs = async () => {
    try {
      const analyses = await apiClient.getAnalyses() as any[]
      setJobs(analyses.map((analysis: any) => ({
        id: analysis.analysis_id,
        name: analysis.project_name || analysis.results?.data_info?.project_name || analysis.analysis_name || `Analysis ${analysis.analysis_id.slice(0, 8)}`,
        status: analysis.status === "queued" ? "running" : analysis.status,
        progress: analysis.progress,
        type: analysis.analysis_types?.join(", ") || "Analysis",
        error: analysis.error
      } as Job)))
    } catch (err) {
      console.error("Failed to load jobs:", err)
    }
  }

  const handleUpload = async (newFiles: any[]) => {
    // Files are already uploaded, just add them to the list
    setFiles(prevFiles => [...newFiles, ...prevFiles])
  }

  const handleCreateAnalysis = async (
    selectedFiles: string[], 
    analysisType: string, 
    analysisName: string,
    preprocessing: {
      normalization: string
      scalingMethod: string
      logTransform: boolean
      logBase: string
      pValue: string
    }
  ) => {
    try {
      // Map analysis type to the correct backend analysis types
      let analysisTypes: string[]
      if (analysisType === "QC") {
        analysisTypes = ["qc"]
      } else if (analysisType === "Full") {
        analysisTypes = ["qc", "pca", "statistical", "ml"]
      } else {
        analysisTypes = [analysisType.toLowerCase()]
      }

      const response = await apiClient.createAnalysis({
        file_ids: selectedFiles,
        analysis_types: analysisTypes,
        project_name: analysisName,
        // Add preprocessing parameters
        scaling_method: preprocessing.scalingMethod,
        log_transform: preprocessing.logTransform,
        log_base: preprocessing.logBase,
        p_value_threshold: parseFloat(preprocessing.pValue)
      }) as any

      // Add new job to the list
      const newJob: Job = {
        id: response.analysis_id,
        name: analysisName,
        status: "running" as const,
        progress: 0,
        type: analysisType
      }

      setJobs(prevJobs => [newJob, ...prevJobs])
      setIsModalOpen(false)
      
      // Refresh jobs after a moment to get updated status
      setTimeout(loadJobs, 1000)
    } catch (err) {
      console.error("Failed to create analysis:", err)
      alert("Failed to create analysis. Please try again.")
    }
  }

  const handleUploadClick = () => {
    // Switch to files tab first
    setActiveTab("files")
    // Then trigger the file input after a short delay to ensure tab is active
    setTimeout(() => {
      document.getElementById("file-upload")?.click()
    }, 100)
  }

  if (loading) {
    return (
      <MainLayout>
        <div className="flex items-center justify-center min-h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      </MainLayout>
    )
  }

  return (
    <MainLayout>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">My Jobs</h1>
        <div className="flex gap-2">
          <Button variant="outline" onClick={loadData}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
          <Button onClick={handleUploadClick}>
            <Upload className="mr-2 h-4 w-4" />
            Upload
          </Button>
          <Button onClick={() => setIsModalOpen(true)} disabled={files.length === 0}>
            <Plus className="mr-2 h-4 w-4" />
            New Analysis
          </Button>
        </div>
      </div>

      {error && (
        <Card className="mb-6 border-destructive">
          <CardContent className="p-4">
            <p className="text-destructive">{error}</p>
          </CardContent>
        </Card>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList>
          <TabsTrigger value="jobs">Active Jobs ({jobs.length})</TabsTrigger>
          <TabsTrigger value="files">Files ({files.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="jobs" className="space-y-4">
          {jobs.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center p-6">
                <p className="text-muted-foreground mb-4">No jobs yet. Upload files and start your first analysis.</p>
                <Button onClick={() => setIsModalOpen(true)} disabled={files.length === 0}>
                  <Plus className="mr-2 h-4 w-4" />
                  New Analysis
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {jobs.map((job) => (
                <JobCard key={job.id} job={job} />
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="files">
          <div className="space-y-4">
            <UploadDropzone onUpload={handleUpload} />

            {files.length === 0 ? (
              <Card>
                <CardContent className="flex flex-col items-center justify-center p-6">
                  <p className="text-muted-foreground">Upload your first mzML or mwTab to get started.</p>
                </CardContent>
              </Card>
            ) : (
              <FileTable files={files} />
            )}
          </div>
        </TabsContent>
      </Tabs>

      <NewAnalysisModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        files={files}
        onCreateAnalysis={handleCreateAnalysis}
      />
    </MainLayout>
  )
}
