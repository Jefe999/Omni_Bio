"use client"

import type React from "react"

import { createContext, useContext, useEffect, useState } from "react"
import { useRouter, usePathname } from "next/navigation"
import { apiClient } from "@/lib/api"

type User = {
  id: string
  name: string
  email: string
  lab?: string
  apiKey: string
}

type AuthContextType = {
  user: User | null
  token: string | null
  apiKey: string | null
  login: (apiKey: string, user?: Partial<User>) => Promise<void>
  logout: () => void
  isLoading: boolean
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  token: null,
  apiKey: null,
  login: async () => {},
  logout: () => {},
  isLoading: true,
})

export const useAuth = () => useContext(AuthContext)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [apiKey, setApiKey] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()
  const pathname = usePathname()

  useEffect(() => {
    // Check for API key in localStorage
    const storedApiKey = localStorage.getItem("omnibio_api_key")
    const storedUser = localStorage.getItem("omnibio_user")

    if (storedApiKey && storedUser) {
      try {
        const userData = JSON.parse(storedUser)
        setApiKey(storedApiKey)
        setUser(userData)
        apiClient.setApiKey(storedApiKey)
      } catch (error) {
        console.error("Failed to parse stored user", error)
        localStorage.removeItem("omnibio_api_key")
        localStorage.removeItem("omnibio_user")
      }
    }

    setIsLoading(false)
  }, [])

  useEffect(() => {
    // Redirect to login if no API key and not already on login page
    if (!isLoading && !apiKey && pathname !== "/login") {
      router.push("/login")
    }

    // Redirect to dashboard if API key exists and on login page
    if (!isLoading && apiKey && pathname === "/login") {
      router.push("/dashboard")
    }
  }, [apiKey, isLoading, pathname, router])

  const login = async (newApiKey: string, userInfo?: Partial<User>) => {
    try {
      // Validate API key with backend
      const isValid = await apiClient.validateApiKey(newApiKey)
      
      if (!isValid) {
        throw new Error("Invalid API key")
      }

      // Create user object (with defaults if not provided)
      const newUser: User = {
        id: userInfo?.id || "user-" + Date.now(),
        name: userInfo?.name || "OmniBio User",
        email: userInfo?.email || "user@omnibio.com",
        lab: userInfo?.lab || "",
        apiKey: newApiKey,
      }

      // Store in localStorage
      localStorage.setItem("omnibio_api_key", newApiKey)
      localStorage.setItem("omnibio_user", JSON.stringify(newUser))
      
      // Update state
      setApiKey(newApiKey)
      setUser(newUser)
      apiClient.setApiKey(newApiKey)
      
      router.push("/dashboard")
    } catch (error) {
      console.error("Login failed:", error)
      throw error
    }
  }

  const logout = () => {
    localStorage.removeItem("omnibio_api_key")
    localStorage.removeItem("omnibio_user")
    setApiKey(null)
    setUser(null)
    apiClient.clearApiKey()
    router.push("/login")
  }

  return (
    <AuthContext.Provider 
      value={{ 
        user, 
        token: apiKey, // For compatibility with existing components
        apiKey, 
        login, 
        logout, 
        isLoading 
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}
