import { Bell, User, Search, Menu } from 'lucide-react';
import { cn } from '../lib/utils';

const Navbar = ({ onMenuClick, className }) => {
  return (
    <nav className={cn(
      "h-16 bg-white border-b border-gray-200 fixed top-0 left-0 right-0 z-50 px-4 lg:px-6",
      className
    )}>
      <div className="h-full flex items-center justify-between max-w-full">
        {/* Left Section */}
        <div className="flex items-center gap-4">
          <button
            onClick={onMenuClick}
            className="lg:hidden p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <Menu className="w-5 h-5 text-gray-600" />
          </button>
          
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-lg">IC</span>
            </div>
            <div className="hidden sm:block">
              <h1 className="text-lg font-bold text-primary">IntelliCredit AI</h1>
              <p className="text-xs text-gray-500">AI Powered Credit Appraisal</p>
            </div>
          </div>
        </div>

        {/* Center - Search */}
        <div className="hidden md:flex flex-1 max-w-md mx-8">
          <div className="relative w-full">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search applications..."
              className="w-full pl-10 pr-4 py-2 bg-gray-50 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
            />
          </div>
        </div>

        {/* Right Section */}
        <div className="flex items-center gap-3">
          <button className="relative p-2 hover:bg-gray-100 rounded-lg transition-colors">
            <Bell className="w-5 h-5 text-gray-600" />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-danger rounded-full"></span>
          </button>
          
          <div className="flex items-center gap-2 pl-3 border-l border-gray-200">
            <div className="w-9 h-9 bg-primary/10 rounded-full flex items-center justify-center">
              <User className="w-5 h-5 text-primary" />
            </div>
            <div className="hidden lg:block">
              <p className="text-sm font-medium text-gray-900">Credit Officer</p>
              <p className="text-xs text-gray-500">Admin</p>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;