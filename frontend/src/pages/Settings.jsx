import React from 'react';
import { Settings as SettingsIcon, Bell, Shield, Key } from 'lucide-react';

export default function Settings() {
  return (
    <div className="flex-1 p-8 overflow-y-auto w-full max-w-7xl mx-auto custom-scrollbar">
      <div className="flex items-center gap-3 mb-8">
        <SettingsIcon className="w-8 h-8 text-emerald-400" />
        <h1 className="text-2xl font-bold tracking-tight text-white">Account Settings</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {/* Settings Navigation */}
        <div className="space-y-2">
          <div className="flex items-center gap-3 px-4 py-3 bg-slate-800 text-emerald-400 rounded-lg cursor-pointer">
             <UserIcon className="w-5 h-5" />
             <span className="font-semibold text-sm">Profile Details</span>
          </div>
          <div className="flex items-center gap-3 px-4 py-3 hover:bg-slate-800/50 text-slate-400 hover:text-slate-200 rounded-lg cursor-pointer transition-colors">
             <Bell className="w-5 h-5" />
             <span className="font-semibold text-sm">Notifications</span>
          </div>
          <div className="flex items-center gap-3 px-4 py-3 hover:bg-slate-800/50 text-slate-400 hover:text-slate-200 rounded-lg cursor-pointer transition-colors">
             <Shield className="w-5 h-5" />
             <span className="font-semibold text-sm">Privacy</span>
          </div>
          <div className="flex items-center gap-3 px-4 py-3 hover:bg-slate-800/50 text-slate-400 hover:text-slate-200 rounded-lg cursor-pointer transition-colors">
             <Key className="w-5 h-5" />
             <span className="font-semibold text-sm">API Keys</span>
          </div>
        </div>
        
        {/* Settings Content */}
        <div className="md:col-span-2 bg-slate-900 border border-slate-800 rounded-2xl p-8">
          <h2 className="text-lg font-bold text-white mb-6">Profile Details</h2>
          
          <div className="space-y-6">
            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Display Name</label>
              <input type="text" defaultValue="Varshith M." className="w-full bg-slate-950 border border-slate-700 text-white rounded-lg px-4 py-2.5 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none transition-all" />
            </div>
            
            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Email Address</label>
              <input type="email" defaultValue="user@example.com" className="w-full bg-slate-950 border border-slate-700 text-white rounded-lg px-4 py-2.5 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none transition-all" />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Bio</label>
              <textarea defaultValue="Fantasy Cricket Machine Learning Enthusiast" rows={4} className="w-full bg-slate-950 border border-slate-700 text-white rounded-lg px-4 py-2.5 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none transition-all resize-none"></textarea>
            </div>

            <div className="pt-4 border-t border-slate-800 flex justify-end">
              <button className="px-6 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold rounded-lg transition-colors shadow-[0_0_15px_rgba(52,211,153,0.3)]">
                Save Changes
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const UserIcon = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
);
