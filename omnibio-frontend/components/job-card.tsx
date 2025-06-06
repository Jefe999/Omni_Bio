"use client"

import { Card, CardContent, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Loader2, CheckCircle, AlertTriangle, Eye } from "lucide-react"
import Link from "next/link"

interface Job {
  id: string
  name: string
  status: "running" | "completed" | "failed"
  progress: number
  type: string
  error?: string
}

interface JobCardProps {
  job: Job
}

export function JobCard({ job }: JobCardProps) {
  return (
    <Card className="overflow-hidden">
      <CardContent className="p-4">
        <div className="flex justify-between items-start mb-2">
          <div>
            <h3 className="font-medium">{job.name}</h3>
            <p className="text-sm text-muted-foreground">Type: {job.type}</p>
          </div>
          <Badge
            variant={job.status === "completed" ? "default" : job.status === "running" ? "outline" : "destructive"}
          >
            {job.status === "running" ? "Running" : job.status === "completed" ? "Completed" : "Failed"}
          </Badge>
        </div>

        {job.status === "running" && (
          <div className="space-y-2 mt-4">
            <div className="flex justify-between text-sm">
              <span>{job.progress}% Complete</span>
              <span className="flex items-center">
                <Loader2 className="h-3 w-3 animate-spin mr-1" />
                Processing
              </span>
            </div>
            <Progress value={job.progress} className="h-2" />
          </div>
        )}

        {job.status === "completed" && (
          <div className="flex items-center text-sm text-green-600 mt-4">
            <CheckCircle className="h-4 w-4 mr-1" />
            Analysis completed successfully
          </div>
        )}

        {job.status === "failed" && (
          <div className="mt-4">
            <div className="flex items-center text-sm text-destructive">
              <AlertTriangle className="h-4 w-4 mr-1" />
              Analysis failed
            </div>
            {job.error && <p className="text-xs text-muted-foreground mt-1">{job.error}</p>}
          </div>
        )}
      </CardContent>

      <CardFooter className="p-4 pt-0 flex justify-end">
        {job.status === "completed" ? (
          <Button asChild size="sm">
            <Link href={`/job/${job.id}`}>
              <Eye className="h-4 w-4 mr-1" />
              View Results
            </Link>
          </Button>
        ) : job.status === "failed" ? (
          <Button variant="outline" size="sm">
            Download Log
          </Button>
        ) : (
          <Button variant="outline" size="sm" disabled>
            <Loader2 className="h-3 w-3 animate-spin mr-1" />
            Processing
          </Button>
        )}
      </CardFooter>
    </Card>
  )
}
