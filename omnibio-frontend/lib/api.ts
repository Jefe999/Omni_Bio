const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

class ApiClient {
  private baseUrl: string
  private apiKey: string | null = null

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl
  }

  setApiKey(apiKey: string) {
    this.apiKey = apiKey
  }

  clearApiKey() {
    this.apiKey = null
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string> || {}),
    }

    // Add API key to headers
    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey
    }

    const config: RequestInit = {
      ...options,
      headers,
    }

    try {
      const response = await fetch(url, config)
      
      if (!response.ok) {
        const errorText = await response.text()
        throw new ApiError(response.status, errorText || `HTTP ${response.status}`)
      }

      // Handle empty responses
      const contentType = response.headers.get('content-type')
      if (contentType && contentType.includes('application/json')) {
        return await response.json()
      } else {
        return {} as T
      }
    } catch (error) {
      if (error instanceof ApiError) {
        throw error
      }
      throw new ApiError(0, `Network error: ${error}`)
    }
  }

  // Health check
  async health() {
    return this.request('/health')
  }

  // Authentication
  async validateApiKey(apiKey: string) {
    const tempClient = new ApiClient(this.baseUrl)
    tempClient.setApiKey(apiKey)
    try {
      await tempClient.request('/health')
      return true
    } catch {
      return false
    }
  }

  // File operations
  async uploadFile(file: File, onProgress?: (progress: number) => void): Promise<any> {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${this.baseUrl}/upload`, {
      method: 'POST',
      headers: {
        'X-API-Key': this.apiKey || '',
      },
      body: formData,
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new ApiError(response.status, errorText || `Upload failed`)
    }

    return await response.json()
  }

  async getFiles() {
    return this.request('/files')
  }

  async deleteFile(fileId: string) {
    return this.request(`/files/${fileId}`, { method: 'DELETE' })
  }

  // Analysis operations
  async createAnalysis(data: {
    file_ids: string[]
    analysis_types: string[]
    project_name: string
    scaling_method?: string
    log_transform?: boolean
    log_base?: string
    p_value_threshold?: number
    group_column?: string
  }) {
    return this.request('/analyze', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async getAnalyses() {
    return this.request('/analyses')
  }

  async getAnalysis(analysisId: string) {
    return this.request(`/analyses/${analysisId}`)
  }

  async getAnalysisFiles(analysisId: string) {
    return this.request(`/analyses/${analysisId}/files`)
  }

  async getAnalysisResults(analysisId: string) {
    return this.request(`/analyses/${analysisId}/results`)
  }

  // Download operations
  async downloadResults(analysisId: string, format: 'json' | 'csv' = 'json') {
    const response = await fetch(`${this.baseUrl}/analyses/${analysisId}/download/${format}`, {
      headers: {
        'X-API-Key': this.apiKey || '',
      },
    })

    if (!response.ok) {
      throw new ApiError(response.status, 'Download failed')
    }

    return response.blob()
  }
}

// Create singleton instance
export const apiClient = new ApiClient(API_BASE_URL)

// Export types for use in components
export type FileInfo = {
  id: string
  name: string
  type: string
  date: string
  size: string
  file_id?: string
  original_filename?: string
  file_type?: string
  uploaded_at?: string
  size_bytes?: number
}

export type JobInfo = {
  id: string
  name?: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  progress: number
  type?: string
  error?: string
  analysis_id?: string
  started_at?: string
  completed_at?: string
  message?: string
}

export type AnalysisRequest = {
  file_ids: string[]
  analysis_types: string[]
  project_name: string
  scaling_method?: string
  log_transform?: boolean
  log_base?: string
  p_value_threshold?: number
  group_column?: string
} 