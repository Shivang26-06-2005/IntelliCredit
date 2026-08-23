import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { cn } from '../lib/utils';

const RiskScoreCard = ({ score, decision, loanLimit, interestRate }) => {
  const getScoreColor = (score) => {
    if (score >= 80) return 'text-accent';
    if (score >= 60) return 'text-warning';
    return 'text-danger';
  };

  const getScoreBg = (score) => {
    if (score >= 80) return 'bg-accent/10';
    if (score >= 60) return 'bg-warning/10';
    return 'bg-danger/10';
  };

  const getDecisionBadge = (decision) => {
    const styles = {
      'Approved': 'bg-accent/10 text-accent border-accent/20',
      'Conditional Approval': 'bg-warning/10 text-warning border-warning/20',
      'Rejected': 'bg-danger/10 text-danger border-danger/20'
    };
    return styles[decision] || 'bg-gray-100 text-gray-600 border-gray-200';
  };

  // Calculate gauge position (0-100 mapped to rotation)
  const gaugeRotation = -90 + (score * 1.8); // -90 to 90 degrees

  return (
    <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-6">
      <h3 className="text-lg font-semibold text-primary mb-6">Risk Assessment</h3>
      
      <div className="flex flex-col items-center">
        {/* Speedometer Gauge */}
        <div className="relative w-48 h-24 mb-4">
          {/* Background arc */}
          <svg viewBox="0 0 200 100" className="w-full h-full">
            {/* Background track */}
            <path
              d="M 20 100 A 80 80 0 0 1 180 100"
              fill="none"
              stroke="#e5e7eb"
              strokeWidth="20"
              strokeLinecap="round"
            />
            {/* Color segments */}
            <path
              d="M 20 100 A 80 80 0 0 1 70 30.7"
              fill="none"
              stroke="#EF4444"
              strokeWidth="20"
              strokeLinecap="round"
            />
            <path
              d="M 70 30.7 A 80 80 0 0 1 130 30.7"
              fill="none"
              stroke="#F59E0B"
              strokeWidth="20"
              strokeLinecap="round"
            />
            <path
              d="M 130 30.7 A 80 80 0 0 1 180 100"
              fill="none"
              stroke="#22C55E"
              strokeWidth="20"
              strokeLinecap="round"
            />
          </svg>
          
          {/* Needle */}
          <div 
            className="absolute bottom-0 left-1/2 w-1 h-20 bg-primary origin-bottom transition-transform duration-1000 ease-out"
            style={{ 
              transform: `translateX(-50%) rotate(${gaugeRotation}deg)`,
              transformOrigin: 'bottom center'
            }}
          >
            <div className="absolute -top-1 -left-1.5 w-4 h-4 bg-primary rounded-full"></div>
          </div>
          
          {/* Center pivot */}
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 w-4 h-4 bg-primary rounded-full border-2 border-white"></div>
        </div>

        {/* Score Display */}
        <div className={cn(
          "text-center px-6 py-3 rounded-xl mb-4",
          getScoreBg(score)
        )}>
          <span className={cn("text-4xl font-bold", getScoreColor(score))}>
            {score}
          </span>
          <span className="text-gray-500 text-sm ml-1">/100</span>
        </div>

        {/* Decision Badge */}
        <span className={cn(
          "px-4 py-2 rounded-full text-sm font-medium border mb-6",
          getDecisionBadge(decision)
        )}>
          {decision}
        </span>

        {/* Loan Details */}
        <div className="w-full grid grid-cols-2 gap-4 pt-4 border-t border-gray-100">
          <div className="text-center">
            <p className="text-xs text-gray-500 mb-1">Loan Limit</p>
            <p className="text-lg font-semibold text-primary">{loanLimit}</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500 mb-1">Interest Rate</p>
            <p className="text-lg font-semibold text-primary">{interestRate}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskScoreCard;
