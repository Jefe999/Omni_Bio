"use client"

import type React from "react"

import { useState } from "react"
import MainLayout from "@/components/main-layout"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useAuth } from "@/components/auth-provider"
import { Copy, Plus, Trash } from "lucide-react"
import { toast } from "@/components/ui/use-toast"
import { Toaster } from "@/components/ui/toaster"

// Mock API keys
const mockApiKeys = [
  { id: "key_1", key: "key_1234abcd5678efgh", expires: "2025-12-31" },
  { id: "key_2", key: "key_5678ijkl9012mnop", expires: "2025-06-30" },
]

export default function SettingsPage() {
  const { user } = useAuth()
  const [apiKeys, setApiKeys] = useState(mockApiKeys)
  const [currentPassword, setCurrentPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")

  const handleGenerateKey = () => {
    // In a real app, this would call an API to generate a new key
    const newKey = {
      id: `key_${Math.random().toString(36).substring(2, 10)}`,
      key: `key_${Math.random().toString(36).substring(2, 10)}${Math.random().toString(36).substring(2, 10)}`,
      expires: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString().split("T")[0],
    }

    setApiKeys([newKey, ...apiKeys])
    toast({
      title: "API Key Generated",
      description: "Your new API key has been generated successfully.",
    })
  }

  const handleRevokeKey = (keyId: string) => {
    // In a real app, this would call an API to revoke the key
    setApiKeys(apiKeys.filter((key) => key.id !== keyId))
    toast({
      title: "API Key Revoked",
      description: "The API key has been revoked successfully.",
    })
  }

  const handleCopyKey = (key: string) => {
    navigator.clipboard.writeText(key)
    toast({
      title: "API Key Copied",
      description: "The API key has been copied to your clipboard.",
    })
  }

  const handleChangePassword = (e: React.FormEvent) => {
    e.preventDefault()

    // Validate passwords
    if (newPassword !== confirmPassword) {
      toast({
        title: "Error",
        description: "New passwords do not match.",
        variant: "destructive",
      })
      return
    }

    // In a real app, this would call an API to change the password
    toast({
      title: "Password Changed",
      description: "Your password has been changed successfully.",
    })

    // Reset form
    setCurrentPassword("")
    setNewPassword("")
    setConfirmPassword("")
  }

  return (
    <MainLayout>
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Settings</h1>
      </div>

      <Tabs defaultValue="profile" className="w-full">
        <TabsList className="w-full justify-start">
          <TabsTrigger value="profile">Profile</TabsTrigger>
          <TabsTrigger value="api-keys">API Keys</TabsTrigger>
          <TabsTrigger value="password">Password</TabsTrigger>
        </TabsList>

        <TabsContent value="profile" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Profile Information</CardTitle>
              <CardDescription>View and update your profile information.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">Name</Label>
                <Input id="name" defaultValue={user?.name} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input id="email" type="email" defaultValue={user?.email} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="lab">Lab</Label>
                <Input id="lab" defaultValue={user?.lab || ""} />
              </div>
            </CardContent>
            <CardFooter>
              <Button>Save Changes</Button>
            </CardFooter>
          </Card>
        </TabsContent>

        <TabsContent value="api-keys" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>API Keys</CardTitle>
              <CardDescription>Manage your API keys for programmatic access to the platform.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button onClick={handleGenerateKey}>
                <Plus className="mr-2 h-4 w-4" />
                Generate New Key
              </Button>

              {apiKeys.length === 0 ? (
                <p className="text-sm text-muted-foreground">No API keys found. Generate a new key to get started.</p>
              ) : (
                <div className="space-y-4">
                  <div className="rounded-md border">
                    <div className="grid grid-cols-3 p-4 font-medium border-b">
                      <div>API Key</div>
                      <div>Expires</div>
                      <div className="text-right">Actions</div>
                    </div>
                    {apiKeys.map((key) => (
                      <div key={key.id} className="grid grid-cols-3 p-4 items-center">
                        <div className="font-mono text-sm">
                          {key.key.substring(0, 8)}...{key.key.substring(key.key.length - 4)}
                        </div>
                        <div>{key.expires}</div>
                        <div className="flex justify-end gap-2">
                          <Button variant="ghost" size="sm" onClick={() => handleCopyKey(key.key)}>
                            <Copy className="h-4 w-4" />
                            <span className="sr-only">Copy</span>
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleRevokeKey(key.id)}
                            className="text-destructive hover:text-destructive"
                          >
                            <Trash className="h-4 w-4" />
                            <span className="sr-only">Revoke</span>
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="password" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Change Password</CardTitle>
              <CardDescription>Update your password to keep your account secure.</CardDescription>
            </CardHeader>
            <form onSubmit={handleChangePassword}>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="current-password">Current Password</Label>
                  <Input
                    id="current-password"
                    type="password"
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="new-password">New Password</Label>
                  <Input
                    id="new-password"
                    type="password"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="confirm-password">Confirm New Password</Label>
                  <Input
                    id="confirm-password"
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    required
                  />
                </div>
              </CardContent>
              <CardFooter>
                <Button type="submit">Change Password</Button>
              </CardFooter>
            </form>
          </Card>
        </TabsContent>
      </Tabs>

      <Toaster />
    </MainLayout>
  )
}
