import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, X, FileText, FileSpreadsheet, Scale, Users, Landmark, PieChart, CheckCircle, Loader2 } from 'lucide-react';
import { cn } from '../lib/utils';

const documentTypes = [
  { id: 'alm', label: 'ALM (Asset-Liability Management)', icon: Scale, description: 'Asset-Liability statements' },
  { id: 'shareholding', label: 'Shareholding Pattern', icon: Users, description: 'Ownership structure details' },
  { id: 'borrowing', label: 'Borrowing Profile', icon: Landmark, description: 'Existing loan and debt details' },
  { id: 'annual_report', label: 'Annual Report (P&L, Cashflow, BS)', icon: FileText, description: 'Financial statements' },
  { id: 'portfolio', label: 'Portfolio Cuts / Performance Data', icon: PieChart, description: 'Performance metrics' },
];

const FileUpload = ({ files, onFilesChange }) => {
  const onDrop = useCallback((acceptedFiles) => {
    const newFiles = acceptedFiles.map(file => ({
      file,
      id: Math.random().toString(36).substr(2, 9),
      type: documentTypes[0].id,
      progress: 100
    }));
    onFilesChange([...files, ...newFiles]);
  }, [files, onFilesChange]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'text/csv': ['.csv']
    }
  });

  const removeFile = (id) => {
    onFilesChange(files.filter(f => f.id !== id));
  };

  const updateFileType = (id, type) => {
    onFilesChange(files.map(f => f.id === id ? { ...f, type } : f));
  };

  return (
    <div className="space-y-4">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={cn(
          "border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all",
          isDragActive 
            ? "border-primary bg-primary/5" 
            : "border-gray-300 hover:border-primary/50 hover:bg-gray-50"
        )}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-3">
          <div className="w-14 h-14 bg-primary/10 rounded-full flex items-center justify-center">
            <Upload className="w-7 h-7 text-primary" />
          </div>
          <div>
            <p className="text-lg font-medium text-gray-900">
              {isDragActive ? 'Drop files here' : 'Drag & drop files here'}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              or click to browse from your computer
            </p>
          </div>
          <p className="text-xs text-gray-400">
            Supported: PDF, Excel, CSV (Max 50MB each)
          </p>
        </div>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-700">
            Uploaded Files ({files.length})
          </h4>
          <div className="space-y-2">
            {files.map((fileObj) => {
              const FileIcon = documentTypes.find(t => t.id === fileObj.type)?.icon || File;
              return (
                <div 
                  key={fileObj.id}
                  className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-100"
                >
                  <div className="w-10 h-10 bg-white rounded-lg flex items-center justify-center shadow-sm">
                    <FileIcon className="w-5 h-5 text-primary" />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {fileObj.file.name}
                    </p>
                    <p className="text-xs text-gray-500">
                      {(fileObj.file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>

                  <select
                    value={fileObj.type}
                    onChange={(e) => updateFileType(fileObj.id, e.target.value)}
                    className="text-xs border border-gray-200 rounded-md px-2 py-1.5 bg-white focus:outline-none focus:ring-2 focus:ring-primary/20"
                  >
                    {documentTypes.map(type => (
                      <option key={type.id} value={type.id}>{type.label}</option>
                    ))}
                  </select>

                  <button
                    onClick={() => removeFile(fileObj.id)}
                    className="p-1.5 hover:bg-gray-200 rounded-md transition-colors"
                  >
                    <X className="w-4 h-4 text-gray-500" />
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Document Types Legend */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-2 pt-4 border-t border-gray-100">
        {documentTypes.map((type) => {
          const Icon = type.icon;
          return (
            <div key={type.id} className="flex items-center gap-2 text-xs text-gray-600">
              <Icon className="w-3.5 h-3.5 text-gray-400" />
              <span>{type.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default FileUpload;
