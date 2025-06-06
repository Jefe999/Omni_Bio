"use client"

import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Upload, CheckCircle, AlertCircle } from "lucide-react"
import { apiClient } from "@/lib/api"

interface UploadDropzoneProps {
  onUpload: (files: any[]) => void
}

export function UploadDropzone({ onUpload }: UploadDropzoneProps) {
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return

      setUploading(true)
      setProgress(0)
      setError(null)
      setSuccess(false)

      try {
        const uploadedFiles = []
        
        for (let i = 0; i < acceptedFiles.length; i++) {
          const file = acceptedFiles[i]
          setProgress(((i / acceptedFiles.length) * 100))
          
          try {
            const result = await apiClient.uploadFile(file)
            uploadedFiles.push({
              id: result.file_id,
              name: result.filename,
              type: result.file_type,
              date: new Date().toISOString().split("T")[0],
              size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
              file_id: result.file_id,
              original_filename: result.filename,
              file_type: result.file_type,
              size_bytes: file.size
            })
          } catch (fileError) {
            console.error(`Failed to upload ${file.name}:`, fileError)
            throw new Error(`Failed to upload ${file.name}: ${fileError}`)
          }
        }
        
        setProgress(100)
        setSuccess(true)
        onUpload(uploadedFiles)
        
        // Reset success state after 3 seconds
        setTimeout(() => setSuccess(false), 3000)
        
      } catch (err) {
        setError(err instanceof Error ? err.message : "Upload failed")
      } finally {
        setUploading(false)
      }
    },
    [onUpload],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/octet-stream": [".mzML"],
      "text/tab-separated-values": [".mwTab", ".txt"],
      "application/json": [".json"],
    },
  })

  return (
    <Card className={`border-2 border-dashed ${isDragActive ? "border-primary bg-primary/5" : "border-border"}`}>
      <CardContent className="p-6">
        <div
          {...getRootProps()}
          className="flex flex-col items-center justify-center space-y-2 text-center cursor-pointer py-4"
        >
          <input {...getInputProps()} id="file-upload" />

          {uploading ? (
            <div className="w-full space-y-4">
              <div className="flex items-center justify-center">
                <p className="text-sm font-medium">Uploading files...</p>
              </div>
              <Progress value={progress} className="h-2 w-full" />
            </div>
          ) : success ? (
            <div className="flex flex-col items-center justify-center space-y-2">
              <CheckCircle className="h-8 w-8 text-green-500" />
              <p className="text-sm font-medium">Files uploaded successfully!</p>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center space-y-2">
              <AlertCircle className="h-8 w-8 text-destructive" />
              <p className="text-sm font-medium text-destructive">{error}</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center space-y-2">
              <Upload className="h-8 w-8 text-muted-foreground" />
              <p className="text-sm font-medium">
                {isDragActive ? "Drop the files here..." : "Drag & drop files here, or click to select files"}
              </p>
              <p className="text-xs text-muted-foreground">Supports mzML, mwTab (.txt), and JSON files</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
