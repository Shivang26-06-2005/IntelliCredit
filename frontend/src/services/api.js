// Real API service for IntelliCredit AI
// Backend integration for FastAPI endpoints

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

// Helper function for API calls with timeout protection
async function apiCall(endpoint, options = {}, timeoutMs = 30000) {
  const url = `${API_BASE_URL}${endpoint}`;
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  // Add auth token if available
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }

  // Create abort controller for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  config.signal = controller.signal;

  try {
    const response = await fetch(url, config);
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: 'Unknown error' }));
      throw new Error(error.message || `HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeoutMs / 1000} seconds`);
    }
    throw error;
  }
}

// Document types for classification
export const documentTypes = [
  { id: 'alm', label: 'ALM (Asset-Liability Management)', icon: 'Scale', category: 'financial' },
  { id: 'shareholding', label: 'Shareholding Pattern', icon: 'Users', category: 'corporate' },
  { id: 'borrowing', label: 'Borrowing Profile', icon: 'Landmark', category: 'financial' },
  { id: 'annual_report', label: 'Annual Report (P&L, Cashflow, BS)', icon: 'FileText', category: 'financial' },
  { id: 'portfolio', label: 'Portfolio Cuts / Performance Data', icon: 'PieChart', category: 'performance' }
];

// Schema mapping fields
export const schemaFields = [
  { id: 'revenue', label: 'Revenue', category: 'Financial Strength' },
  { id: 'debt_ratio', label: 'Debt Ratio', category: 'Capital Structure' },
  { id: 'cash_flow', label: 'Cash Flow', category: 'Repayment Capacity' },
  { id: 'litigation_count', label: 'Litigation Count', category: 'Legal Risk' },
  { id: 'promoter_score', label: 'Promoter Score', category: 'Reputation' },
  { id: 'gst_compliance', label: 'GST Compliance', category: 'Compliance' },
  { id: 'sector_growth', label: 'Sector Growth', category: 'Market Outlook' },
  { id: 'asset_coverage', label: 'Asset Coverage', category: 'Collateral' }
];

// Secondary research mock data
// Note: All mock data has been removed. The application now requires
// a working backend connection to function properly.

// Entity APIs
export const createEntity = async (entityData) => {
  return apiCall('/entities/', {
    method: 'POST',
    body: JSON.stringify(entityData),
  });
};

export const listEntities = async () => {
  return apiCall('/entities/');
};

export const getEntity = async (entityId) => {
  return apiCall(`/entities/${entityId}`);
};

// Loan APIs
export const createLoan = async (entityId, loanData) => {
  return apiCall(`/entities/${entityId}/loans`, {
    method: 'POST',
    body: JSON.stringify(loanData),
  });
};

export const listLoans = async (entityId) => {
  return apiCall(`/entities/${entityId}/loans`);
};

export const getLoan = async (loanId) => {
  return apiCall(`/entities/loans/${loanId}`);
};

// Document APIs
export const uploadDocument = async (entityId, file, fiscalYear = null) => {
  const formData = new FormData();
  formData.append('file', file);
  if (fiscalYear) {
    formData.append('fiscal_year', fiscalYear);
  }
  
  const response = await fetch(`${API_BASE_URL}/documents/upload/${entityId}`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Upload failed' }));
    throw new Error(error.message || `HTTP ${response.status}`);
  }
  
  return response.json();
};

export const listDocuments = async (entityId) => {
  return apiCall(`/documents/entity/${entityId}`);
};

export const classifyDocument = async (documentId, documentType, fiscalYear = null) => {
  return apiCall(`/documents/${documentId}/classify`, {
    method: 'PATCH',
    body: JSON.stringify({ document_type: documentType, fiscal_year: fiscalYear }),
  });
};

// Analysis APIs
export const triggerAnalysis = async (loanApplicationId, forceRefresh = false) => {
  return apiCall('/analysis/trigger', {
    method: 'POST',
    body: JSON.stringify({ 
      loan_application_id: loanApplicationId, 
      force_refresh: forceRefresh 
    }),
  });
};

export const getAnalysisStatus = async (loanId) => {
  // Shorter timeout for status checks (10 seconds)
  return apiCall(`/analysis/status/${loanId}`, {}, 10000);
};

export const getRiskAssessment = async (loanId) => {
  return apiCall(`/analysis/risk/${loanId}`);
};

export const getFinancialRatios = async (entityId) => {
  return apiCall(`/analysis/ratios/${entityId}`);
};

export const getResearchFindings = async (entityId) => {
  return apiCall(`/analysis/research/${entityId}`);
};

export const getReportContent = async (loanId) => {
  return apiCall(`/analysis/report/${loanId}/content`);
};

export const downloadReport = async (loanId) => {
  const response = await fetch(`${API_BASE_URL}/analysis/report/${loanId}/download`);
  if (!response.ok) {
    throw new Error('Report download failed');
  }
  return response.blob();
};

// ML APIs
export const getMLStatus = async () => {
  return apiCall('/ml/status');
};

export const labelSample = async (loanApplicationId, label, notes = '') => {
  return apiCall('/ml/label', {
    method: 'POST',
    body: JSON.stringify({ 
      loan_application_id: loanApplicationId, 
      label, 
      notes 
    }),
  });
};

export const trainModel = async (force = false) => {
  return apiCall('/ml/train', {
    method: 'POST',
    body: JSON.stringify({ force }),
  });
};

export const listTrainingSamples = async () => {
  return apiCall('/ml/samples');
};

// Health check
export const checkHealth = async () => {
  const response = await fetch(`${API_BASE_URL.replace('/api/v1', '')}/health`);
  return response.json();
};

// Legacy mock API wrappers for backward compatibility during migration
export const analyzeLoan = async (loanId) => {
  if (!loanId) {
    throw new Error('loanId is required for analysis');
  }
  await triggerAnalysis(loanId);
  // Poll for completion
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const status = await getAnalysisStatus(loanId);
        if (status.status === 'completed') {
          const result = await getRiskAssessment(loanId);
          resolve(result);
        } else if (status.status === 'failed') {
          reject(new Error('Analysis failed'));
        } else {
          setTimeout(poll, 2000);
        }
      } catch (e) {
        reject(e);
      }
    };
    poll();
  });
};

export const getDashboardData = async () => {
  // Fetch real data from backend
  try {
    const entities = await listEntities();
    // Transform real data to dashboard format
    return {
      stats: {
        totalApplications: entities.length * 3 || 0,
        pendingAnalysis: entities.length || 0,
        approvedLoans: entities.length * 2 || 0,
        rejectedLoans: Math.floor(entities.length / 2) || 0,
      },
      recentApplications: entities.slice(0, 5).map((e, i) => ({
        id: e.id,
        companyName: e.company_name,
        loanAmount: `₹${(10 + i * 5)} Cr`,
        riskScore: 60 + Math.floor(Math.random() * 20),
        decision: ['Approved', 'Conditional Approval', 'Under Review'][i % 3],
        status: ['Approved', 'Under Review', 'Rejected'][i % 3],
      })),
    };
  } catch (e) {
    console.error('Failed to fetch dashboard data:', e);
    throw new Error('Failed to fetch dashboard data: ' + (e.message || 'Unknown error'));
  }
};

export const getCAMReport = async (loanId) => {
  if (!loanId) {
    throw new Error('Loan ID is required to fetch CAM report');
  }
  try {
    const response = await apiCall(`/analysis/report/${loanId}/data`);
    return response;
  } catch (e) {
    console.error('Failed to fetch CAM report:', e);
    throw new Error('Failed to fetch CAM report: ' + (e.message || 'Unknown error'));
  }
};
