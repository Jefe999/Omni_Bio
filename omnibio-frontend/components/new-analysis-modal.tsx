"use client"

import { useState } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface File {
  id: string
  name: string
  type: string
  date: string
  size: string
}

interface NewAnalysisModalProps {
  isOpen: boolean
  onClose: () => void
  files: File[]
  onCreateAnalysis: (selectedFiles: string[], analysisType: string, analysisName: string, preprocessing: { normalization: string; scalingMethod: string; logTransform: boolean; logBase: string; pValue: string }) => void
}

export function NewAnalysisModal({ isOpen, onClose, files, onCreateAnalysis }: NewAnalysisModalProps) {
  const [step, setStep] = useState(1)
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [analysisName, setAnalysisName] = useState("")
  const [analysisType, setAnalysisType] = useState("Full")
  const [normalization, setNormalization] = useState("total_ion_current")
  const [scalingMethod, setScalingMethod] = useState("pareto")
  const [logTransform, setLogTransform] = useState(false)
  const [logBase, setLogBase] = useState("log10")
  const [pValue, setPValue] = useState("0.05")

  // Generate default analysis name
  const generateDefaultName = () => {
    const date = new Date().toISOString().split('T')[0]
    return `Biomarker Analysis ${date}`
  }

  const handleFileSelect = (fileId: string, checked: boolean) => {
    if (checked) {
      setSelectedFiles([...selectedFiles, fileId])
    } else {
      setSelectedFiles(selectedFiles.filter((id) => id !== fileId))
    }
  }

  const handleNext = () => {
    setStep(step + 1)
  }

  const handleBack = () => {
    setStep(step - 1)
  }

  const handleSubmit = () => {
    const finalName = analysisName.trim() || generateDefaultName()
    onCreateAnalysis(selectedFiles, analysisType, finalName, {
      normalization,
      scalingMethod,
      logTransform,
      logBase,
      pValue
    })
    resetForm()
  }

  const resetForm = () => {
    setStep(1)
    setSelectedFiles([])
    setAnalysisName("")
    setAnalysisType("Full")
    setNormalization("total_ion_current")
    setScalingMethod("pareto")
    setLogTransform(false)
    setLogBase("log10")
    setPValue("0.05")
  }

  const handleClose = () => {
    resetForm()
    onClose()
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>New Analysis</DialogTitle>
          <DialogDescription>Create a new biomarker analysis by selecting files and parameters.</DialogDescription>
        </DialogHeader>

        <Tabs value={`step-${step}`} className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="step-1" disabled>
              Files
            </TabsTrigger>
            <TabsTrigger value="step-2" disabled>
              Pipeline
            </TabsTrigger>
            <TabsTrigger value="step-3" disabled>
              Parameters
            </TabsTrigger>
            <TabsTrigger value="step-4" disabled>
              Confirm
            </TabsTrigger>
          </TabsList>

          <TabsContent value="step-1" className="space-y-4 py-4">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="analysis-name">Analysis Name</Label>
                <Input
                  id="analysis-name"
                  placeholder={generateDefaultName()}
                  value={analysisName}
                  onChange={(e) => setAnalysisName(e.target.value)}
                />
              </div>

              <h3 className="text-sm font-medium">Select files for analysis</h3>

              {files.length === 0 ? (
                <p className="text-sm text-muted-foreground">No files available. Please upload files first.</p>
              ) : (
                <div className="max-h-[250px] overflow-y-auto space-y-2">
                  {files.map((file) => (
                    <div key={file.id} className="flex items-center space-x-2">
                      <Checkbox
                        id={file.id}
                        checked={selectedFiles.includes(file.id)}
                        onCheckedChange={(checked) => handleFileSelect(file.id, checked === true)}
                      />
                      <Label htmlFor={file.id} className="flex-1 cursor-pointer">
                        {file.name}
                        <span className="text-xs text-muted-foreground ml-2">({file.type})</span>
                      </Label>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="step-2" className="space-y-4 py-4">
            <div className="space-y-4">
              <h3 className="text-sm font-medium">Choose analysis pipeline</h3>

              <RadioGroup value={analysisType} onValueChange={setAnalysisType} className="space-y-3">
                <div className="flex items-start space-x-2">
                  <RadioGroupItem value="QC" id="qc" className="mt-1" />
                  <div className="space-y-1">
                    <Label htmlFor="qc" className="font-medium">QC Only</Label>
                    <p className="text-xs text-muted-foreground">Quality control plots and data validation</p>
                  </div>
                </div>
                <div className="flex items-start space-x-2">
                  <RadioGroupItem value="Full" id="full" className="mt-1" />
                  <div className="space-y-1">
                    <Label htmlFor="full" className="font-medium">Full Analysis</Label>
                    <p className="text-xs text-muted-foreground">Complete biomarker discovery: QC + Statistics + Machine Learning</p>
                  </div>
                </div>
              </RadioGroup>
            </div>
          </TabsContent>

          <TabsContent value="step-3" className="space-y-4 py-4">
            <div className="space-y-4">
              <h3 className="text-sm font-medium">Analysis parameters</h3>

              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="normalization">Data normalization</Label>
                  <Select value={normalization} onValueChange={setNormalization}>
                    <SelectTrigger id="normalization">
                      <SelectValue placeholder="Select normalization method" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="total_ion_current">Total Ion Current</SelectItem>
                      <SelectItem value="median">Median</SelectItem>
                      <SelectItem value="quantile">Quantile</SelectItem>
                      <SelectItem value="none">None</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="scaling">Feature scaling method</Label>
                  <Select value={scalingMethod} onValueChange={setScalingMethod}>
                    <SelectTrigger id="scaling">
                      <SelectValue placeholder="Select scaling method" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="pareto">Pareto (metabolomics standard)</SelectItem>
                      <SelectItem value="standard">Standard (z-score)</SelectItem>
                      <SelectItem value="robust">Robust (median-based)</SelectItem>
                      <SelectItem value="minmax">Min-Max [0,1]</SelectItem>
                      <SelectItem value="power">Power Transformation</SelectItem>
                      <SelectItem value="none">No Scaling</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="log-transform"
                      checked={logTransform}
                      onCheckedChange={(checked) => setLogTransform(checked === true)}
                    />
                    <Label htmlFor="log-transform" className="font-medium">Apply log transformation</Label>
                  </div>
                  
                  {logTransform && (
                    <div className="ml-6 space-y-2">
                      <Label htmlFor="log-base">Log base</Label>
                      <Select value={logBase} onValueChange={setLogBase}>
                        <SelectTrigger id="log-base">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="log10">Log10</SelectItem>
                          <SelectItem value="log2">Log2</SelectItem>
                          <SelectItem value="ln">Natural Log (ln)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="pvalue">Significance threshold (p-value)</Label>
                  <Select value={pValue} onValueChange={setPValue}>
                    <SelectTrigger id="pvalue">
                      <SelectValue placeholder="Select p-value threshold" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0.01">0.01</SelectItem>
                      <SelectItem value="0.05">0.05</SelectItem>
                      <SelectItem value="0.1">0.1</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="step-4" className="space-y-4 py-4">
            <div className="space-y-4">
              <h3 className="text-sm font-medium">Confirm analysis settings</h3>

              <div className="space-y-2 text-sm">
                <p>
                  <span className="font-medium">Analysis name:</span> {analysisName.trim() || generateDefaultName()}
                </p>
                <p>
                  <span className="font-medium">Selected files:</span> {selectedFiles.length} files
                </p>
                <p>
                  <span className="font-medium">Analysis type:</span> {analysisType}
                </p>
                <p>
                  <span className="font-medium">Data normalization:</span> {normalization.replace("_", " ")}
                </p>
                <p>
                  <span className="font-medium">Feature scaling:</span> {scalingMethod}
                </p>
                {logTransform && (
                  <p>
                    <span className="font-medium">Log transformation:</span> {logBase}
                  </p>
                )}
                <p>
                  <span className="font-medium">P-value threshold:</span> {pValue}
                </p>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        <DialogFooter className="flex justify-between">
          <div className="flex gap-2">
            {step > 1 && (
              <Button variant="outline" onClick={handleBack}>
                Back
              </Button>
            )}
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={handleClose}>
              Cancel
            </Button>
            {step < 4 ? (
              <Button onClick={handleNext} disabled={step === 1 && selectedFiles.length === 0}>
                Next
              </Button>
            ) : (
              <Button onClick={handleSubmit}>
                Start Analysis
              </Button>
            )}
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
