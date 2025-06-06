"use client"

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

// Helper function to format feature names for better display
function formatFeatureName(feature: string): string {
  // If it's an m/z value, try to make it more readable
  if (feature.includes('m/z') || feature.match(/^\d+\.\d+$/)) {
    // Extract numeric part if it's just a number
    const match = feature.match(/(\d+\.\d+)/);
    if (match) {
      const mz = parseFloat(match[1]);
      // Common metabolite mappings (simplified for demo)
      const metaboliteMap: { [key: string]: string } = {
        '123.45': 'Glucose',
        '456.78': 'Lactate', 
        '789.01': 'Citrate',
        '234.56': 'Alanine',
        '567.89': 'Glutamate',
        '890.12': 'Creatinine',
        '345.67': 'Urea',
        '678.90': 'Taurine',
        '901.23': 'Choline',
        '432.10': 'Acetate'
      };
      
      // Check if we have a metabolite name for this m/z
      const key = mz.toFixed(2);
      if (metaboliteMap[key]) {
        return `${metaboliteMap[key]} (m/z ${mz.toFixed(1)})`;
      }
      
      // Otherwise return formatted m/z
      return `m/z ${mz.toFixed(1)}`;
    }
  }
  
  // If it already looks like a metabolite name, return as-is
  if (!feature.includes('m/z') && !feature.match(/^\d+\.\d+$/)) {
    return feature;
  }
  
  return feature;
}

interface Feature {
  id: number
  mz: string
  log2FC: number
  pValue: number
  rocWeight: number
}

interface TopFeatureTableProps {
  features: Feature[]
}

export function TopFeatureTable({ features }: TopFeatureTableProps) {
  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Feature</TableHead>
            <TableHead>Log2 Fold Change</TableHead>
            <TableHead>p-value</TableHead>
            <TableHead>ROC Weight</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {features.map((feature) => (
            <TableRow key={feature.id}>
              <TableCell className="font-medium">{formatFeatureName(feature.mz)}</TableCell>
              <TableCell className={feature.log2FC > 0 ? "text-green-600" : "text-red-600"}>
                {feature.log2FC > 0 ? "+" : ""}
                {feature.log2FC.toFixed(2)}
              </TableCell>
              <TableCell>{feature.pValue.toExponential(2)}</TableCell>
              <TableCell>{feature.rocWeight.toFixed(2)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
