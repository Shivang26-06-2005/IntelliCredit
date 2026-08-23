import { FileText, Clock, CheckCircle, XCircle } from 'lucide-react';
import { cn } from '../lib/utils';

const statCards = [
  {
    title: 'Total Applications',
    value: '156',
    icon: FileText,
    color: 'bg-blue-500',
    trend: '+12%',
    trendUp: true
  },
  {
    title: 'Pending Analysis',
    value: '23',
    icon: Clock,
    color: 'bg-warning',
    trend: '+5%',
    trendUp: false
  },
  {
    title: 'Approved Loans',
    value: '98',
    icon: CheckCircle,
    color: 'bg-accent',
    trend: '+18%',
    trendUp: true
  },
  {
    title: 'Rejected Loans',
    value: '35',
    icon: XCircle,
    color: 'bg-danger',
    trend: '-3%',
    trendUp: true
  }
];

const DashboardCards = () => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {statCards.map((card, index) => {
        const Icon = card.icon;
        return (
          <div 
            key={index}
            className="bg-white rounded-xl p-5 border border-gray-100 shadow-sm hover:shadow-md transition-shadow"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm text-gray-500 font-medium">{card.title}</p>
                <h3 className="text-2xl font-bold text-primary mt-1">{card.value}</h3>
              </div>
              <div className={cn("p-2.5 rounded-lg", card.color)}>
                <Icon className="w-5 h-5 text-white" />
              </div>
            </div>
            <div className="mt-3 flex items-center gap-1">
              <span className={cn(
                "text-xs font-medium",
                card.trendUp ? "text-accent" : "text-danger"
              )}>
                {card.trend}
              </span>
              <span className="text-xs text-gray-400">vs last month</span>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default DashboardCards;
