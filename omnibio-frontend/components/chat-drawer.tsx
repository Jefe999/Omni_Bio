"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetFooter } from "@/components/ui/sheet"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Loader2, Send } from "lucide-react"

interface ChatDrawerProps {
  isOpen: boolean
  onClose: () => void
  jobId: string
}

export function ChatDrawer({ isOpen, onClose, jobId }: ChatDrawerProps) {
  const [messages, setMessages] = useState<Array<{ role: "user" | "assistant"; content: string }>>([
    {
      role: "assistant",
      content: "Hello! I can help you analyze the results of your job. What would you like to know?",
    },
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  const handleSendMessage = async () => {
    if (!input.trim()) return

    const userMessage = { role: "user" as const, content: input }
    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    // Simulate API call with a delay
    setTimeout(() => {
      // Mock responses based on keywords
      let response = "I'm analyzing the data from this job. What specific aspect would you like me to explain?"

      if (input.toLowerCase().includes("pathway")) {
        response =
          "Based on the analysis, I found enrichment in several metabolic pathways. The top enriched pathways include glycolysis (p=0.001), TCA cycle (p=0.003), and fatty acid metabolism (p=0.008). Would you like more details on any specific pathway?"
      } else if (input.toLowerCase().includes("biomarker") || input.toLowerCase().includes("feature")) {
        response =
          "The top biomarkers identified in this analysis are m/z 123.4567 (log2FC: 2.34) and m/z 456.7890 (log2FC: -1.87). These features show strong discriminatory power with ROC weights of 0.85 and 0.78 respectively."
      } else if (input.toLowerCase().includes("pca") || input.toLowerCase().includes("cluster")) {
        response =
          "The PCA plot shows clear separation between sample groups along PC1, which explains 45.2% of the variance. There appears to be some batch effect visible in PC2 (28.7% variance), but it doesn't interfere with the biological signal."
      }

      setMessages((prev) => [...prev, { role: "assistant", content: response }])
      setIsLoading(false)
    }, 1500)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <Sheet open={isOpen} onOpenChange={onClose}>
      <SheetContent className="w-full sm:max-w-md flex flex-col p-0">
        <SheetHeader className="px-4 py-2 border-b">
          <SheetTitle>Analysis Chat</SheetTitle>
        </SheetHeader>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message, index) => (
            <div key={index} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`flex gap-2 max-w-[80%] ${message.role === "user" ? "flex-row-reverse" : ""}`}>
                <Avatar className="h-8 w-8">
                  <AvatarFallback className={message.role === "user" ? "bg-primary" : "bg-muted"}>
                    {message.role === "user" ? "U" : "A"}
                  </AvatarFallback>
                </Avatar>
                <div
                  className={`rounded-lg px-3 py-2 ${
                    message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                  }`}
                >
                  {message.content}
                </div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="flex gap-2 max-w-[80%]">
                <Avatar className="h-8 w-8">
                  <AvatarFallback className="bg-muted">A</AvatarFallback>
                </Avatar>
                <div className="rounded-lg px-3 py-2 bg-muted flex items-center">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span className="ml-2">Analyzing...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <SheetFooter className="px-4 py-2 border-t">
          <div className="flex w-full items-center space-x-2">
            <Input
              placeholder="Ask about your analysis results..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              className="flex-1"
            />
            <Button size="icon" onClick={handleSendMessage} disabled={isLoading || !input.trim()}>
              <Send className="h-4 w-4" />
              <span className="sr-only">Send message</span>
            </Button>
          </div>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
