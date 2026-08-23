import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend
} from 'recharts';

const COLORS = ['#22C55E', '#F59E0B', '#EF4444', '#3B82F6', '#8B5CF6'];

export const RiskDistributionChart = () => {
  const chartData = [
    { name: 'Low Risk', value: 45, count: 70 },
    { name: 'Medium Risk', value: 35, count: 55 },
    { name: 'High Risk', value: 20, count: 31 },
  ];

  return (
    <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-6">
      <h3 className="text-lg font-semibold text-primary mb-4">Risk Score Distribution</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="name" 
              tick={{ fill: '#6B7280', fontSize: 12 }}
              axisLine={{ stroke: '#E5E7EB' }}
            />
            <YAxis 
              tick={{ fill: '#6B7280', fontSize: 12 }}
              axisLine={{ stroke: '#E5E7EB' }}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff', 
                border: '1px solid #E5E7EB',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
              }}
            />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export const SectorBreakdownChart = () => {
  const chartData = [
    { name: 'Manufacturing', value: 35 },
    { name: 'IT Services', value: 25 },
    { name: 'Textiles', value: 20 },
    { name: 'Pharma', value: 15 },
    { name: 'Others', value: 5 },
  ];

  return (
    <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-6">
      <h3 className="text-lg font-semibold text-primary mb-4">Sector Breakdown</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={5}
              dataKey="value"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff', 
                border: '1px solid #E5E7EB',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
              }}
            />
            <Legend 
              verticalAlign="bottom" 
              height={36}
              iconType="circle"
              wrapperStyle={{ fontSize: '12px' }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export const RiskBreakdownChart = ({ data }) => {
  const chartData = [
    { name: 'Financial Strength', score: data?.financialStrength || 75 },
    { name: 'Cashflow Stability', score: data?.cashflowStability || 70 },
    { name: 'Litigation Risk', score: data?.litigationRisk || 40 },
    { name: 'Sector Outlook', score: data?.sectorOutlook || 80 },
    { name: 'Promoter Rep.', score: data?.promoterReputation || 60 },
  ];

  return (
    <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-6">
      <h3 className="text-lg font-semibold text-primary mb-4">Risk Factor Breakdown</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" horizontal={false} />
            <XAxis 
              type="number" 
              domain={[0, 100]}
              tick={{ fill: '#6B7280', fontSize: 12 }}
              axisLine={{ stroke: '#E5E7EB' }}
            />
            <YAxis 
              type="category" 
              dataKey="name"
              width={120}
              tick={{ fill: '#6B7280', fontSize: 11 }}
              axisLine={{ stroke: '#E5E7EB' }}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff', 
                border: '1px solid #E5E7EB',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
              }}
            />
            <Bar dataKey="score" radius={[0, 4, 4, 0]} fill="#0A2540" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export const LoanExposureChart = () => {
  const chartData = [
    { month: 'Jan', amount: 45 },
    { month: 'Feb', amount: 52 },
    { month: 'Mar', amount: 48 },
    { month: 'Apr', amount: 61 },
    { month: 'May', amount: 55 },
    { month: 'Jun', amount: 67 },
  ];

  return (
    <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-6">
      <h3 className="text-lg font-semibold text-primary mb-4">Loan Exposure Trend (Cr)</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="month" 
              tick={{ fill: '#6B7280', fontSize: 12 }}
              axisLine={{ stroke: '#E5E7EB' }}
            />
            <YAxis 
              tick={{ fill: '#6B7280', fontSize: 12 }}
              axisLine={{ stroke: '#E5E7EB' }}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff', 
                border: '1px solid #E5E7EB',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
              }}
              formatter={(value) => [`₹${value} Cr`, 'Amount']}
            />
            <Bar dataKey="amount" fill="#1E3A8A" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default { RiskDistributionChart, SectorBreakdownChart, RiskBreakdownChart, LoanExposureChart };
