import React, { useState, useEffect } from 'react';
import { Search, BrainCircuit, Activity, ChevronDown, ChevronUp, User, Lock, Ban, MapPin, Target, Zap, Download } from 'lucide-react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, PieChart, Pie, Cell, Tooltip, BarChart, Bar, CartesianGrid, XAxis, YAxis, Legend } from 'recharts';
import { useSearchParams } from 'react-router-dom';
import { cn } from '../utils';
import { getTeamLogo, getTeamTheme, getTeamLogoFit } from '../utils/logoUtils';

export default function Dashboard() {
  const [searchParams] = useSearchParams();
  const [players, setPlayers] = useState([]);
  const [matchId, setMatchId] = useState(searchParams.get('matchId') || '335982');
  const team1 = searchParams.get('t1') || 'RCB';
  const team2 = searchParams.get('t2') || 'CSK';
  const [matchContext, setMatchContext] = useState({
    venueAvg: 160,
    ppAvg: 45,
    moAvg: 75,
    doAvg: 40
  });

  useEffect(() => {
    const newMatchId = searchParams.get('matchId');
    if (newMatchId) setMatchId(newMatchId);
  }, [searchParams]);
  const [lockedPlayers, setLockedPlayers] = useState([]);
  const [bannedPlayers, setBannedPlayers] = useState([]);
  const [isCalculating, setIsCalculating] = useState(false);
  const [isOptimized, setIsOptimized] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [roleFilter, setRoleFilter] = useState('ALL');
  const [expandedPlayerId, setExpandedPlayerId] = useState(null);
  
  // New State variables
  const [battingFirstTeam, setBattingFirstTeam] = useState('NONE');
  const [pitchType, setPitchType] = useState('Neutral');
  const [weatherCondition, setWeatherCondition] = useState('Clear');
  const [swapModePlayerId, setSwapModePlayerId] = useState(null);
  const [swapError, setSwapError] = useState(null);
  const [modalPlayer, setModalPlayer] = useState(null);
  const [playerStats, setPlayerStats] = useState([]);
  const [h2hStats, setH2hStats] = useState(null);
  const [activeModalTab, setActiveModalTab] = useState('L5');
  const [numLineups, setNumLineups] = useState(1);
  const [generatedLineups, setGeneratedLineups] = useState([]);
  const [currentLineupIndex, setCurrentLineupIndex] = useState(0);

  useEffect(() => {
    fetch(`http://localhost:8000/players/${matchId}`)
      .then(res => res.json())
      .then(data => {
         setPlayers(data);
         setLockedPlayers([]);
         setBannedPlayers([]);
         setIsOptimized(false);
      })
      .catch(console.error);

    fetch(`http://localhost:8000/match_context/${matchId}`)
      .then(res => res.json())
      .then(data => {
        if (data && data.venueAvg) {
          setMatchContext(data);
        }
      })
      .catch(console.error);
  }, [matchId]);

  const optimalPlayers = players.filter(p => p.isOptimal);
  const creditsUsed = optimalPlayers.reduce((sum, p) => sum + p.credits, 0);
  const totalPoints = optimalPlayers.reduce((sum, p) => sum + p.proj_points, 0);

  const stats = {
    credits: isOptimized ? creditsUsed : 0,
    points: isOptimized ? totalPoints : 0,
  };

  const roles = ['ALL', 'WK', 'BAT', 'AR', 'BOWL'];

  const filteredPlayers = players.filter((p) => {
    const matchesSearch = p.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesRole = roleFilter === 'ALL' || p.role === roleFilter;
    return matchesSearch && matchesRole;
  });

  const t1Abbr = team1.split(' ').map(t => t[0]).join('').substring(0,3).toUpperCase();
  const sortedPlayers = [...filteredPlayers].sort((a, b) => {
    const aIsT1 = a.team === t1Abbr || a.team === team1 || a.team === team1.substring(0, 3).toUpperCase();
    const bIsT1 = b.team === t1Abbr || b.team === team1 || b.team === team1.substring(0, 3).toUpperCase();
    
    if (aIsT1 && !bIsT1) return -1;
    if (!aIsT1 && bIsT1) return 1;
    return b.proj_points - a.proj_points;
  });

  const toggleLock = (e, id) => {
    e.stopPropagation();
    if (lockedPlayers.includes(id)) {
       setLockedPlayers(lockedPlayers.filter(pId => pId !== id));
    } else {
       setLockedPlayers([...lockedPlayers, id]);
       setBannedPlayers(bannedPlayers.filter(pId => pId !== id)); // Mutual Exclusion
    }
  };

  const toggleBan = (e, id) => {
    e.stopPropagation();
    if (bannedPlayers.includes(id)) {
       setBannedPlayers(bannedPlayers.filter(pId => pId !== id));
    } else {
       setBannedPlayers([...bannedPlayers, id]);
       setLockedPlayers(lockedPlayers.filter(pId => pId !== id)); // Mutual Exclusion
    }
  };

  const handleOptimize = async () => {
    setIsCalculating(true);
    setIsOptimized(false);
    
    try {
      const res = await fetch('http://localhost:8000/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          match_id: matchId,
          locked_player_ids: lockedPlayers,
          banned_player_ids: bannedPlayers,
          batting_first_team: battingFirstTeam === 'NONE' ? null : battingFirstTeam,
          pitch_type: pitchType,
          weather_condition: weatherCondition,
          num_lineups: parseInt(numLineups, 10)
        })
      });
      const data = await res.json();
      
      if (data.error) {
        alert(data.error);
        setIsCalculating(false);
        return;
      }
      
      if (data.lineups && data.lineups.length > 0) {
        setGeneratedLineups(data.lineups);
        setCurrentLineupIndex(0);
        setPlayers(data.lineups[0]);
        setIsOptimized(true);
      }
    } catch (e) {
      console.error(e);
      alert("Failed to connect to ML Backend.");
    } finally {
      setIsCalculating(false);
    }
  };

  const exportToCsv = () => {
    if (!generatedLineups || generatedLineups.length === 0) return;
    
    let csvContent = "Lineup_Number,Player_1,Player_2,Player_3,Player_4,Player_5,Player_6,Player_7,Player_8,Player_9,Player_10,Player_11\n";
    
    generatedLineups.forEach((lineup, index) => {
      const optimal = lineup.filter(p => p.isOptimal);
      const sorted = [...optimal].sort((a,b) => {
          const roleOrder = { 'WK': 1, 'BAT': 2, 'AR': 3, 'BOWL': 4 };
          return roleOrder[a.role] - roleOrder[b.role];
      });
      const names = sorted.map(p => `"${p.name} (${p.role})"`);
      csvContent += `${index + 1},${names.join(",")}\n`;
    });
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `optimizer_lineups_${team1}_vs_${team2}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const teamData = [];
  const uniqueTeams = [...new Set(optimalPlayers.map(p => p.team))];
  uniqueTeams.forEach((team) => {
    teamData.push({
      name: team,
      value: optimalPlayers.filter(p => p.team === team).length,
      color: getTeamTheme(team).primary
    });
  });

  const roleData = [
    { name: 'BAT', value: optimalPlayers.filter(p => p.role === 'BAT').length, color: '#3b82f6' },
    { name: 'BOWL', value: optimalPlayers.filter(p => p.role === 'BOWL').length, color: '#f97316' },
    { name: 'AR', value: optimalPlayers.filter(p => p.role === 'AR').length, color: '#a855f7' },
    { name: 'WK', value: optimalPlayers.filter(p => p.role === 'WK').length, color: '#ec4899' },
  ].filter(d => d.value > 0);

  const handleSwap = (newPlayer) => {
    if (!swapModePlayerId) return;
    if (newPlayer.isOptimal) return;

    const currentOptimals = players.filter(p => p.isOptimal && p.id !== swapModePlayerId);
    const currentCredits = currentOptimals.reduce((sum, p) => sum + p.credits, 0);
    
    if (currentCredits + newPlayer.credits > 100) {
      setSwapError(`Cannot afford ${newPlayer.name}. Exceeds 100 max credits.`);
      setTimeout(() => setSwapError(null), 3000);
      return;
    }
    
    const proposedOptimals = [...currentOptimals, newPlayer];
    const teamCounts = {};
    proposedOptimals.forEach(p => teamCounts[p.team] = (teamCounts[p.team] || 0) + 1);
    if (Object.values(teamCounts).some(c => c > 7)) {
       setSwapError(`Exceeds maximum 7 players per franchise limit.`);
       setTimeout(() => setSwapError(null), 3000);
       return;
    }

    const wkCount = proposedOptimals.filter(p => p.role === 'WK').length;
    const batCount = proposedOptimals.filter(p => p.role === 'BAT').length;
    const arCount = proposedOptimals.filter(p => p.role === 'AR').length;
    const bowlCount = proposedOptimals.filter(p => p.role === 'BOWL').length;
    
    if (wkCount < 1 || wkCount > 4 || batCount < 3 || batCount > 6 || arCount < 1 || arCount > 4 || bowlCount < 3 || bowlCount > 6) {
       setSwapError(`Violates structural rules (WK: ${wkCount}/1-4, BAT: ${batCount}/3-6, AR: ${arCount}/1-4, BOWL: ${bowlCount}/3-6).`);
       setTimeout(() => setSwapError(null), 3000);
       return;
    }

    setPlayers(prev => prev.map(p => {
      if (p.id === swapModePlayerId) return { ...p, isOptimal: false };
      if (p.id === newPlayer.id) return { ...p, isOptimal: true };
      return p;
    }));
    setSwapModePlayerId(null);
  };

  const openPlayerModal = (e, player) => {
    e.stopPropagation();
    setModalPlayer(player);
    setPlayerStats([]);
    setH2hStats(null);
    setActiveModalTab('L5');
    fetch(`http://localhost:8000/player_stats/${matchId}/${player.name}`)
      .then(res => res.json())
      .then(data => setPlayerStats(data))
      .catch(console.error);
    fetch(`http://localhost:8000/matchup/${matchId}/${player.name}`)
      .then(res => res.json())
      .then(data => {
         if (!data.error) setH2hStats(data);
      })
      .catch(console.error);
  };

  return (
    <div className="flex flex-col flex-1 overflow-hidden font-sans relative">
      {/* Dashboard Sub-Header */}
      <div className="flex items-center justify-between px-6 py-4 bg-slate-900 border-b border-slate-800 shadow-md z-10 shrink-0">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div 
              className="w-10 h-10 rounded-full border flex items-center justify-center overflow-hidden shadow-sm shrink-0 bg-white relative"
              style={{ borderColor: getTeamTheme(team1).primary }}
            >
               <img src={getTeamLogo(team1)} alt={team1} className={`w-full h-full ${getTeamLogoFit(team1)}`} />
            </div>
            <span className="text-[10px] font-black text-slate-600 bg-slate-950 px-1.5 py-0.5 rounded shadow-inner border border-slate-800/50">VS</span>
            <div 
              className="w-10 h-10 rounded-full border flex items-center justify-center overflow-hidden shadow-sm shrink-0 bg-white relative"
              style={{ borderColor: getTeamTheme(team2).primary }}
            >
               <img src={getTeamLogo(team2)} alt={team2} className={`w-full h-full ${getTeamLogoFit(team2)}`} />
            </div>
          </div>
          <div className="pl-3 border-l border-slate-800/50">
            <h1 className="text-xl font-bold tracking-tight text-white">{team1} vs {team2}</h1>
            <p className="text-sm text-slate-400">Match Optimizer</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-end gap-3 lg:gap-6">
          <div className="flex items-center gap-2 bg-slate-900 border border-slate-700/50 p-1 rounded-lg">
            <select value={pitchType} onChange={(e) => setPitchType(e.target.value)} className="bg-slate-950/80 border border-slate-700 text-slate-300 text-xs font-semibold uppercase tracking-wider rounded-md px-3 py-1.5 hover:border-cyan-500/50 outline-none cursor-pointer">
              <option value="Neutral">🌱 Neutral Pitch</option>
              <option value="Spin">🌪️ Spin Pitch</option>
              <option value="Pace">⚡ Pace Pitch</option>
            </select>
            <select value={weatherCondition} onChange={(e) => setWeatherCondition(e.target.value)} className="bg-slate-950/80 border border-slate-700 text-slate-300 text-xs font-semibold uppercase tracking-wider rounded-md px-3 py-1.5 hover:border-cyan-500/50 outline-none cursor-pointer">
              <option value="Clear">☀️ Clear Weather</option>
              <option value="Overcast">☁️ Overcast</option>
              <option value="Dew">💧 Heavy Dew</option>
            </select>
          </div>

          <select
            value={battingFirstTeam}
            onChange={(e) => setBattingFirstTeam(e.target.value)}
            className="bg-slate-950 border border-slate-700 text-slate-300 text-xs font-semibold uppercase tracking-wider rounded-lg px-4 py-2 hover:border-cyan-500/50 outline-none focus:ring-1 focus:ring-cyan-500/50 appearance-none cursor-pointer hidden md:block"
          >
            <option value="NONE">⚖️ Neutral Toss</option>
            <option value={team1}>🪙 {team1} Bats 1st</option>
            <option value={team2}>🪙 {team2} Bats 1st</option>
          </select>
          
          <select value={numLineups} onChange={(e) => setNumLineups(e.target.value)} className="bg-slate-950 border border-slate-700 text-slate-300 text-xs font-semibold uppercase tracking-wider rounded-lg px-3 py-2 hover:border-cyan-500/50 outline-none focus:ring-1 focus:ring-cyan-500/50 cursor-pointer hidden lg:block">
            <option value={1}>1 Lineup</option>
            <option value={5}>5 Lineups</option>
            <option value={10}>10 Lineups</option>
            <option value={20}>20 Lineups</option>
          </select>

          <div className="flex items-center gap-6 bg-slate-950 px-6 py-2 rounded-full border border-slate-800">
            <div className="flex flex-col">
              <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">Credits Used</span>
              <span className={cn("text-lg font-bold font-mono transition-colors duration-500", isOptimized ? "text-cyan-400" : "text-white")}>
                {stats.credits.toFixed(1)} <span className="text-slate-500 text-sm">/ 100</span>
              </span>
            </div>
            <div className="w-px h-8 bg-slate-800"></div>
            <div className="flex flex-col">
              <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">Projected Points</span>
              <span className={cn("text-lg font-bold font-mono transition-colors duration-500", isOptimized ? "text-emerald-400" : "text-white")}>
                {stats.points.toFixed(1)}
              </span>
            </div>
          </div>

          <button
            onClick={handleOptimize}
            disabled={isCalculating}
            className={cn(
              "relative group px-6 py-3 rounded-lg font-semibold flex items-center gap-2 transition-all duration-300 overflow-hidden",
              isCalculating 
                ? "bg-slate-800 text-slate-400 cursor-not-allowed" 
                : "bg-emerald-500 hover:bg-emerald-400 text-slate-950 hover:shadow-[0_0_20px_rgba(52,211,153,0.4)]"
            )}
          >
            {isCalculating ? (
              <>
                <div className="absolute inset-0 bg-emerald-400/20 animate-pulse"></div>
                <Activity className="w-5 h-5 animate-spin" />
                <span>Optimizing...</span>
              </>
            ) : (
              <>
                <BrainCircuit className="w-5 h-5" />
                <span>Run ML Optimizer</span>
              </>
            )}
          </button>

          {isOptimized && generatedLineups.length > 0 && (
            <button
              onClick={exportToCsv}
              className="px-4 py-3 rounded-lg font-bold flex items-center gap-2 transition-all duration-300 border bg-indigo-500/10 text-indigo-400 border-indigo-500/30 hover:bg-indigo-500/20 hover:border-indigo-400"
              title="Export all generated lineups to CSV"
            >
              <Download className="w-4 h-4" />
              <span className="hidden xl:inline">Export CSV</span>
            </button>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Pane - Player Pool (40%) */}
        <div className="w-[40%] flex flex-col bg-slate-900 border-r border-slate-800">
          
          {/* Match Context HUD */}
          <div className="p-4 border-b border-slate-800 bg-slate-900/50">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Match Context</h3>
            <div className="grid grid-cols-4 gap-3">
              <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 flex flex-col">
                <div className="flex items-center gap-1.5 mb-1 text-cyan-400">
                  <Target className="w-3.5 h-3.5" />
                  <span className="text-[10px] font-medium uppercase tracking-wider">Venue Avg</span>
                </div>
                <span className="text-lg font-bold text-slate-200">{matchContext.venueAvg}</span>
              </div>
              <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 flex flex-col">
                <div className="flex items-center gap-1.5 mb-1 text-amber-400">
                  <Activity className="w-3.5 h-3.5" />
                  <span className="text-[10px] font-medium uppercase tracking-wider">PP Avg</span>
                </div>
                <span className="text-sm font-bold text-slate-200 mt-1">{matchContext.ppAvg}</span>
              </div>
              <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 flex flex-col">
                 <div className="flex items-center gap-1.5 mb-1 text-indigo-400">
                  <Activity className="w-3.5 h-3.5" />
                  <span className="text-[10px] font-medium uppercase tracking-wider">MO Avg</span>
                </div>
                <span className="text-sm font-bold text-slate-200 mt-1">{matchContext.moAvg}</span>
              </div>
              <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 flex flex-col">
                 <div className="flex items-center gap-1.5 mb-1 text-red-400">
                  <Activity className="w-3.5 h-3.5" />
                  <span className="text-[10px] font-medium uppercase tracking-wider">Death Avg</span>
                </div>
                <span className="text-sm font-bold text-slate-200 mt-1">{matchContext.doAvg}</span>
              </div>
            </div>
          </div>

          <div className="p-4 border-b border-slate-800">
            <div className="relative mb-4">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
              <input
                type="text"
                placeholder="Search players..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 text-white rounded-lg pl-10 pr-4 py-2.5 outline-none focus:border-cyan-400/50 focus:ring-1 focus:ring-cyan-400/50 transition-all placeholder:text-slate-600"
              />
            </div>
            <div className="flex gap-2">
              {roles.map(role => (
                <button
                  key={role}
                  onClick={() => setRoleFilter(role)}
                  className={cn(
                    "px-4 py-1.5 rounded-full text-xs font-medium transition-all duration-200",
                    roleFilter === role 
                      ? "bg-cyan-400/20 text-cyan-400 border border-cyan-400/30" 
                      : "bg-slate-800 text-slate-400 hover:bg-slate-700 border border-transparent"
                  )}
                >
                  {role}
                </button>
              ))}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto custom-scrollbar">
            <table className="w-full text-left border-collapse">
              <thead className="sticky top-0 bg-slate-900/95 backdrop-blur z-10 border-b border-slate-800">
                <tr>
                  <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Player</th>
                  <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Role</th>
                  <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider text-right">Credits</th>
                  <th className="px-4 py-3 text-xs font-semibold text-emerald-400 uppercase tracking-wider text-right">Proj Pts</th>
                  <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider text-right">Constraints</th>
                  <th className="px-4 py-3 w-10"></th>
                </tr>
              </thead>
              <tbody>
                {sortedPlayers.map(player => (
                  <React.Fragment key={player.id}>
                    <tr 
                      className={cn(
                        "group transition-colors border-b border-slate-800/50",
                        expandedPlayerId === player.id ? "bg-slate-800/50" : "hover:bg-slate-800/50",
                        isOptimized && player.isOptimal && "bg-emerald-400/5",
                        bannedPlayers.includes(player.id) && "opacity-50 grayscale",
                        swapModePlayerId && !player.isOptimal ? "cursor-pointer hover:bg-cyan-900/30 inset-0 relative hover:border-cyan-500 shadow-[inset_0_0_20px_rgba(34,211,238,0)] hover:shadow-[inset_0_0_20px_rgba(34,211,238,0.2)]" : "cursor-pointer",
                        swapModePlayerId === player.id && "bg-cyan-900/40 border-cyan-500/50 "
                      )}
                      onClick={(e) => {
                         if (swapModePlayerId && !player.isOptimal) {
                            handleSwap(player);
                         } else {
                            setExpandedPlayerId(expandedPlayerId === player.id ? null : player.id);
                         }
                      }}
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-3">
                          <div 
                            className="w-8 h-8 rounded-full border bg-white flex items-center justify-center overflow-hidden shrink-0 transition-colors relative shadow-sm"
                            style={{ borderColor: getTeamTheme(player.team).primary }}
                          >
                            <img src={getTeamLogo(player.team)} alt={player.team} className={`w-full h-full ${getTeamLogoFit(player.team)} opacity-90 group-hover:opacity-100 transition-opacity`} />
                          </div>
                          <span className={cn(
                            "font-medium truncate max-w-[120px]",
                            isOptimized && player.isOptimal ? "text-emerald-400" : "text-slate-200",
                            lockedPlayers.includes(player.id) && "text-amber-400"
                          )}>
                            {player.name}
                          </span>
                          {player.form_tag === 'hot' && (
                             <span className="text-[10px] ml-1 px-1.5 py-0.5 rounded-full bg-orange-500/20 text-orange-400 border border-orange-500/30 flex items-center shrink-0 shadow-inner max-w-fit" title="In form (>40 runs or >=2 wkts in prev match)">
                               🔥 Hot
                             </span>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-xs font-medium px-2 py-1 bg-slate-800 text-slate-300 rounded text-center min-w-[3rem] inline-block">
                          {player.role}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-slate-300">{player.credits}</td>
                      <td className="px-4 py-3 text-right font-mono text-emerald-400 font-semibold">{player.proj_points.toFixed(1)}</td>
                      <td className="px-4 py-3 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <button 
                            onClick={(e) => toggleLock(e, player.id)}
                            className={cn(
                              "p-1.5 rounded-md transition-all duration-200",
                              lockedPlayers.includes(player.id) ? "bg-amber-500/20 text-amber-400 border border-amber-500/30" : "hover:bg-slate-700 text-slate-500 hover:text-amber-400/70 border border-transparent"
                            )}
                            title="Lock player into lineup"
                          >
                            <Lock className="w-4 h-4" />
                          </button>
                          <button 
                            onClick={(e) => toggleBan(e, player.id)}
                            className={cn(
                              "p-1.5 rounded-md transition-all duration-200",
                              bannedPlayers.includes(player.id) ? "bg-red-500/20 text-red-400 border border-red-500/30" : "hover:bg-slate-700 text-slate-500 hover:text-red-400/70 border border-transparent"
                            )}
                            title="Ban player from lineup"
                          >
                            <Ban className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-center">
                        {expandedPlayerId === player.id ? (
                          <ChevronUp className="w-5 h-5 text-slate-500 group-hover:text-cyan-400" />
                        ) : (
                          <ChevronDown className="w-5 h-5 text-slate-500 group-hover:text-cyan-400" />
                        )}
                      </td>
                    </tr>
                    {expandedPlayerId === player.id && (
                      <tr>
                        <td colSpan={6} className="bg-slate-950 p-6 border-b border-slate-800">
                          <div className="flex bg-slate-900 rounded-xl border border-slate-800 p-4">
                            <div className="flex-1">
                              <h4 className="text-sm font-semibold text-slate-300 mb-4 flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <Activity className="w-4 h-4 text-cyan-400" />
                                  ML Feature Breakdown
                                </div>
                                <button onClick={(e) => openPlayerModal(e, player)} className="text-xs bg-slate-800 hover:bg-slate-700 text-cyan-400 px-3 py-1.5 rounded-md transition-colors flex items-center gap-1.5 shadow-md border border-slate-700">
                                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"></path></svg>
                                  Deep Dive Analytics
                                </button>
                              </h4>
                              <div className="space-y-3">
                                {player.radar.map(stat => (
                                  <div key={stat.phase}>
                                    <div className="flex justify-between text-xs mb-1">
                                      <span className="text-slate-400">{stat.phase === 'PP' ? 'Powerplay' : stat.phase === 'MO' ? 'Middle Overs' : 'Death Overs'} Percentile</span>
                                      <span className="text-cyan-400 font-mono">{stat.val}%</span>
                                    </div>
                                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                                      <div 
                                        className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full"
                                        style={{ width: `${stat.val}%` }}
                                      ></div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                            <div className="w-[200px] h-[160px]">
                              <ResponsiveContainer width="100%" height="100%">
                                <RadarChart cx="50%" cy="50%" outerRadius="70%" data={player.radar}>
                                  <PolarGrid stroke="#1e293b" />
                                  <PolarAngleAxis dataKey="phase" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                                  <Radar name="Player" dataKey="val" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.3} />
                                </RadarChart>
                              </ResponsiveContainer>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right Pane - The Pitch & Analytics (60%) */}
        <div className="w-[60%] bg-slate-950 flex flex-col p-6 items-center relative overflow-y-auto custom-scrollbar">
          
          <h2 className="text-lg font-bold text-slate-300 mb-6 flex items-center gap-2 shrink-0">
            Optimal Lineup Projection
            {isOptimized && <span className="text-xs font-normal px-2 py-0.5 bg-emerald-400/10 text-emerald-400 border border-emerald-400/20 rounded-full ml-2">Active</span>}
          </h2>

          {isOptimized && generatedLineups.length > 1 && (
            <div className="flex items-center gap-4 mb-4 bg-slate-900 border border-slate-800 rounded-lg px-4 py-2 shrink-0 z-10 w-full max-w-xl justify-between shadow-md">
              <button 
                onClick={() => {
                  const newIdx = Math.max(0, currentLineupIndex - 1);
                  setCurrentLineupIndex(newIdx);
                  setPlayers(generatedLineups[newIdx]);
                }} 
                disabled={currentLineupIndex === 0}
                className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed rounded text-slate-300 text-sm font-bold transition-colors"
              >
                &larr; Prev
              </button>
              <div className="text-cyan-400 font-mono font-bold text-sm flex items-center gap-2">
                Lineup {currentLineupIndex + 1} <span className="text-slate-500">/ {generatedLineups.length}</span>
              </div>
              <button 
                onClick={() => {
                  const newIdx = Math.min(generatedLineups.length - 1, currentLineupIndex + 1);
                  setCurrentLineupIndex(newIdx);
                  setPlayers(generatedLineups[newIdx]);
                }} 
                disabled={currentLineupIndex === generatedLineups.length - 1}
                className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed rounded text-slate-300 text-sm font-bold transition-colors"
              >
                Next &rarr;
              </button>
            </div>
          )}

          {/* Rule Tracker HUD */}
          <div className="flex bg-slate-900 border border-slate-800 rounded-full px-6 py-2 shadow-xl shrink-0 mb-6 gap-6 items-center z-10 relative">
             <div className="flex items-center gap-1.5">
               <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">WK</span>
               <span className={cn("text-xs font-mono font-bold transition-colors", isOptimized && optimalPlayers.filter(p=>p.role==='WK').length > 0 ? "text-cyan-400" : "text-white")}>
                 {optimalPlayers.filter(p=>p.role==='WK').length}<span className="text-slate-600">/4</span>
               </span>
             </div>
             <div className="w-px h-3 bg-slate-800"></div>
             <div className="flex items-center gap-1.5">
               <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">BAT</span>
               <span className={cn("text-xs font-mono font-bold transition-colors", isOptimized && optimalPlayers.filter(p=>p.role==='BAT').length >= 3 ? "text-cyan-400" : "text-white")}>
                 {optimalPlayers.filter(p=>p.role==='BAT').length}<span className="text-slate-600">/6</span>
               </span>
             </div>
             <div className="w-px h-3 bg-slate-800"></div>
             <div className="flex items-center gap-1.5">
               <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">AR</span>
               <span className={cn("text-xs font-mono font-bold transition-colors", isOptimized && optimalPlayers.filter(p=>p.role==='AR').length > 0 ? "text-cyan-400" : "text-white")}>
                 {optimalPlayers.filter(p=>p.role==='AR').length}<span className="text-slate-600">/4</span>
               </span>
             </div>
             <div className="w-px h-3 bg-slate-800"></div>
             <div className="flex items-center gap-1.5">
               <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">BOWL</span>
               <span className={cn("text-xs font-mono font-bold transition-colors", isOptimized && optimalPlayers.filter(p=>p.role==='BOWL').length >= 3 ? "text-cyan-400" : "text-white")}>
                 {optimalPlayers.filter(p=>p.role==='BOWL').length}<span className="text-slate-600">/6</span>
               </span>
             </div>
          </div>

          <div className="relative w-full max-w-xl aspect-[3/4] shrink-0 bg-green-900 border-4 border-white rounded-[100px] overflow-hidden shadow-2xl flex flex-col justify-between py-12 px-8 z-0">
            {/* Pitch pattern overlay */}
            <div className="absolute inset-0 opacity-20 pointer-events-none" style={{
              backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 40px, rgba(0,0,0,0.1) 40px, rgba(0,0,0,0.1) 80px)'
            }}></div>
            
            {/* Inner ring */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 border-2 border-white/30 rounded-full pointer-events-none"></div>
            
            {/* 30 yard circle */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[80%] h-[60%] border-2 border-dashed border-white/20 rounded-[100px] pointer-events-none"></div>

            {/* Scanning Animation */}
            {isCalculating && (
              <div className="absolute inset-0 bg-cyan-500/10 z-20 pointer-events-none overflow-hidden rounded-[100px]">
                <div className="absolute w-full h-1 bg-cyan-400 shadow-[0_0_20px_rgba(34,211,238,1)] animate-scan"></div>
                <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCI+CjxyZWN0IHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJyZ2JhKDM0LCAyMTEsIDIzOCwgMC4xKSIgc3Ryb2tlLXdpZHRoPSIxIi8+Cjwvc3ZnPg==')] opacity-30"></div>
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-cyan-400 font-mono font-bold tracking-widest bg-slate-950/80 px-4 py-2 rounded-lg border border-cyan-400/50 backdrop-blur">
                  OPTIMIZING MATRIX...
                </div>
              </div>
            )}

            {/* Swap Overlays */}
            {swapError && (
              <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-red-500/90 text-white px-4 py-2 rounded-lg text-sm font-bold z-50 shadow-xl flex items-center gap-2 max-w-[90%] text-center border border-red-400 backdrop-blur">
                 <span>⚠️</span> {swapError}
              </div>
            )}
            {swapModePlayerId && (
              <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-cyan-950/90 text-cyan-400 border border-cyan-400 px-4 py-2 rounded-lg text-sm font-bold z-50 shadow-[0_0_20px_rgba(34,211,238,0.2)] flex items-center gap-2 hover:bg-cyan-900/90 cursor-pointer transition-colors backdrop-blur whitespace-nowrap" onClick={() => setSwapModePlayerId(null)}>
                 <span className="animate-spin text-[10px]">⚽</span> SELECT PLAYER FROM POOL TO SWAP... (CLICK TO CANCEL)
              </div>
            )}

            {/* Players Layout */}
            {isOptimized ? (
              <div className="absolute inset-0 z-10 transition-opacity duration-300" style={{ opacity: isCalculating ? 0.3 : 1 }}>
                {/* Wicket Keeper */}
                <PlayerNode player={optimalPlayers.find(p => p.role === 'WK')} top="8%" left="50%" onSwapClick={(p) => setSwapModePlayerId(p.id)} swapSourceActive={swapModePlayerId === optimalPlayers.find(p => p.role === 'WK')?.id} />
                
                {/* Bowlers */}
                <div className="absolute w-full top-[25%] flex justify-around px-8">
                  {optimalPlayers.filter(p => p.role === 'BOWL').map((p, i) => (
                    <PlayerNode key={p.id} player={p} pos="relative" onSwapClick={(p) => setSwapModePlayerId(p.id)} swapSourceActive={swapModePlayerId === p.id} />
                  ))}
                </div>

                {/* All Rounders */}
                <div className="absolute w-full top-[50%] flex justify-around px-16">
                  {optimalPlayers.filter(p => p.role === 'AR').map((p, i) => (
                    <PlayerNode key={p.id} player={p} pos="relative" onSwapClick={(p) => setSwapModePlayerId(p.id)} swapSourceActive={swapModePlayerId === p.id} />
                  ))}
                </div>

                {/* Batsmen */}
                <div className="absolute w-full top-[75%] flex justify-around px-12">
                  {optimalPlayers.filter(p => p.role === 'BAT').map((p, i) => (
                    <PlayerNode key={p.id} player={p} pos="relative" onSwapClick={(p) => setSwapModePlayerId(p.id)} swapSourceActive={swapModePlayerId === p.id} />
                  ))}
                </div>
              </div>
            ) : (
               <div className="absolute inset-0 z-10 flex flex-col justify-between py-12">
                 {/* Placeholders */}
                 <div className="w-full flex justify-center"><PlaceholderNode role="1 WK" /></div>
                 <div className="w-full flex justify-around px-8">
                   <PlaceholderNode role="BOWL" />
                   <PlaceholderNode role="BOWL" />
                   <PlaceholderNode role="BOWL" />
                 </div>
                 <div className="w-full flex justify-around px-16">
                   <PlaceholderNode role="AR" />
                   <PlaceholderNode role="AR" />
                   <PlaceholderNode role="AR" />
                 </div>
                 <div className="w-full flex justify-around px-12">
                   <PlaceholderNode role="BAT" />
                   <PlaceholderNode role="BAT" />
                   <PlaceholderNode role="BAT" />
                   <PlaceholderNode role="BAT" />
                 </div>
               </div>
            )}

          </div>

          {/* NEW: Lineup Analytics (Visible after optimization) */}
          {isOptimized && (
            <div className="w-full max-w-xl mt-8 shrink-0 bg-slate-900 border border-slate-800 rounded-xl p-5 mb-8">
              <h3 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                Lineup Analytics
              </h3>
              <div className="grid grid-cols-2 gap-4 h-[220px]">
                
                {/* Team Split Donut */}
                <div className="flex flex-col items-center bg-slate-950 rounded-lg py-2 border border-slate-800/50">
                  <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Team Split</h4>
                  <div className="flex-1 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={teamData}
                          cx="50%"
                          cy="50%"
                          innerRadius={50}
                          outerRadius={70}
                          paddingAngle={5}
                          dataKey="value"
                          stroke="none"
                        >
                          {teamData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px' }}
                          itemStyle={{ color: '#e2e8f0', fontSize: '12px', fontWeight: 'bold' }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="flex gap-4 mt-2">
                    {teamData.map(d => (
                      <div key={d.name} className="flex items-center gap-1.5">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d.color }}></div>
                        <span className="text-xs text-slate-300 font-medium">{d.name} ({d.value})</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Role Distribution Donut */}
                <div className="flex flex-col items-center bg-slate-950 rounded-lg py-2 border border-slate-800/50">
                  <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Role Distribution</h4>
                  <div className="flex-1 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={roleData}
                          cx="50%"
                          cy="50%"
                          innerRadius={50}
                          outerRadius={70}
                          paddingAngle={5}
                          dataKey="value"
                          stroke="none"
                        >
                          {roleData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px' }}
                          itemStyle={{ color: '#e2e8f0', fontSize: '12px', fontWeight: 'bold' }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="flex gap-3 justify-center flex-wrap px-2 mt-2">
                    {roleData.map(d => (
                      <div key={d.name} className="flex items-center gap-1.5">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d.color }}></div>
                        <span className="text-xs text-slate-300 font-medium">{d.name} ({d.value})</span>
                      </div>
                    ))}
                  </div>
                </div>

              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Advanced Analytics Player Modal */}
      {modalPlayer && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-slate-950/90 backdrop-blur-md" onClick={() => setModalPlayer(null)}>
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-4xl shadow-[0_0_50px_rgba(0,0,0,0.5)] overflow-hidden flex flex-col max-h-[90vh]" onClick={e => e.stopPropagation()}>
             {/* Modal Header */}
             <div className="flex items-center justify-between p-6 border-b border-slate-800 bg-slate-950/50">
                <div className="flex items-center gap-5">
                   <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-full border bg-white flex items-center justify-center overflow-hidden shrink-0 shadow-[0_0_20px_rgba(255,255,255,0.1)] relative" style={{ borderColor: getTeamTheme(modalPlayer.team).primary }}>
                      <img src={getTeamLogo(modalPlayer.team)} alt={modalPlayer.team} className={`w-full h-full ${getTeamLogoFit(modalPlayer.team)}`} />
                   </div>
                   <div>
                      <h2 className="text-2xl sm:text-3xl font-bold text-white flex items-center gap-3">
                         {modalPlayer.name}
                         {modalPlayer.form_tag === 'hot' && <span className="text-xs px-2.5 py-1 rounded-full bg-orange-500/20 text-orange-400 border border-orange-500/30 font-semibold shadow-inner shadow-orange-500/10">🔥 Hot Form</span>}
                      </h2>
                      <div className="flex items-center gap-3 mt-2 text-sm text-slate-400">
                         <span className="bg-slate-800 border border-slate-700 px-2 py-0.5 rounded text-white font-bold">{modalPlayer.role}</span>
                         <span className="w-1.5 h-1.5 rounded-full bg-slate-700"></span>
                         <span className="font-mono text-cyan-400 font-semibold">{modalPlayer.credits} CR</span>
                         <span className="w-1.5 h-1.5 rounded-full bg-slate-700"></span>
                         <span className="font-mono text-emerald-400 font-semibold">{modalPlayer.proj_points.toFixed(1)} Projected Pts</span>
                      </div>
                   </div>
                </div>
                <button onClick={() => setModalPlayer(null)} className="p-2.5 bg-slate-800 hover:bg-slate-700 hover:text-white text-slate-400 rounded-full transition-colors border border-slate-700">
                   <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M18 6L6 18M6 6l12 12"></path></svg>
                </button>
             </div>
             
             {/* Modal Content */}
             <div className="p-6 overflow-y-auto custom-scrollbar flex flex-col gap-6 bg-gradient-to-b from-slate-900 to-slate-950 flex-1">
                <div className="flex border-b border-slate-800 pb-2 gap-4 shrink-0 px-2 mt-[-10px]">
                    <button onClick={() => setActiveModalTab('L5')} className={cn("text-xs font-bold uppercase tracking-wider transition-colors pb-2 border-b-2", activeModalTab === 'L5' ? "text-cyan-400 border-cyan-400" : "text-slate-500 border-transparent hover:text-slate-300")}>
                        Recent Form (L5)
                    </button>
                    <button onClick={() => setActiveModalTab('H2H')} className={cn("text-xs font-bold uppercase tracking-wider transition-colors pb-2 border-b-2 flex items-center gap-1.5", activeModalTab === 'H2H' ? "text-indigo-400 border-indigo-400" : "text-slate-500 border-transparent hover:text-slate-300")}>
                        <Zap className="w-3 h-3" /> H2H vs Opponent
                    </button>
                </div>
                
                {activeModalTab === 'H2H' ? (
                   !h2hStats ? (
                      <div className="h-64 flex items-center justify-center text-slate-500 flex-col gap-4">
                        <Activity className="w-8 h-8 animate-spin text-indigo-500" />
                        <p className="font-medium tracking-wide">Fetching Franchise Matchup History...</p>
                      </div>
                   ) : (
                      <div className="flex flex-col gap-6">
                        <div className="bg-indigo-500/10 border border-indigo-500/20 p-5 rounded-xl flex items-center justify-between shadow-inner">
                           <div>
                              <h3 className="text-xl font-bold text-white">{modalPlayer.name} vs {h2hStats.opponent}</h3>
                              <p className="text-sm text-indigo-300 mt-1">Historically played <span className="font-bold text-white">{h2hStats.matches_played}</span> matches against this franchise.</p>
                           </div>
                           <div className="w-12 h-12 rounded-full border bg-white flex items-center justify-center overflow-hidden shadow-sm" style={{ borderColor: getTeamTheme(h2hStats.opponent).primary }}>
                             <img src={getTeamLogo(h2hStats.opponent)} alt={h2hStats.opponent} className={`w-full h-full ${getTeamLogoFit(h2hStats.opponent)}`} />
                           </div>
                        </div>
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                           <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl flex flex-col items-center justify-center">
                              <span className="text-xs text-slate-500 uppercase font-black tracking-widest mb-1.5 text-center">Total Runs</span>
                              <span className="text-3xl font-black text-rose-400 font-mono drop-shadow-md">{h2hStats.total_runs}</span>
                           </div>
                           <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl flex flex-col items-center justify-center">
                              <span className="text-xs text-slate-500 uppercase font-black tracking-widest mb-1.5 text-center">Strike Rate</span>
                              <span className="text-3xl font-black text-cyan-400 font-mono drop-shadow-md">{h2hStats.strike_rate}</span>
                           </div>
                           <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl flex flex-col items-center justify-center">
                              <span className="text-xs text-slate-500 uppercase font-black tracking-widest mb-1.5 text-center">Total Wickets</span>
                              <span className="text-3xl font-black text-amber-400 font-mono drop-shadow-md">{h2hStats.total_wickets}</span>
                           </div>
                           <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl flex flex-col items-center justify-center">
                              <span className="text-xs text-slate-500 uppercase font-black tracking-widest mb-1.5 text-center">Avg Fantasy Pts</span>
                              <span className="text-3xl font-black text-emerald-400 font-mono drop-shadow-md">{h2hStats.avg_fantasy_points}</span>
                           </div>
                        </div>
                        
                        {/* Micro Matchups Table */}
                        {h2hStats.micro_matchups && h2hStats.micro_matchups.length > 0 && (
                           <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-inner w-full mt-2">
                              <h4 className="text-sm font-bold text-slate-300 p-4 border-b border-slate-800 flex items-center gap-2">
                                 <Zap className="w-4 h-4 text-cyan-400" /> Micro-Matchups vs Tonight's Bowlers ({h2hStats.opponent})
                              </h4>
                              <div className="overflow-x-auto w-full custom-scrollbar">
                                 <table className="w-full text-left whitespace-nowrap">
                                    <thead className="bg-slate-950/50">
                                       <tr>
                                          <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Bowler</th>
                                          <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider text-right">Runs</th>
                                          <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider text-right">Balls</th>
                                          <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider text-right">SR</th>
                                          <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider text-right">Outs</th>
                                       </tr>
                                    </thead>
                                    <tbody className="divide-y divide-slate-800/50">
                                       {h2hStats.micro_matchups.sort((a,b) => b.runs - a.runs).map((m, i) => (
                                          <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                                             <td className="px-4 py-3 text-sm font-bold text-slate-200">{m.bowler}</td>
                                             <td className="px-4 py-3 text-sm text-right text-rose-400 font-mono font-bold animate-pulse-slow">{m.runs}</td>
                                             <td className="px-4 py-3 text-sm text-right text-slate-400 font-mono">{m.balls}</td>
                                             <td className="px-4 py-3 text-sm text-right text-cyan-400 font-mono">{m.sr}%</td>
                                             <td className="px-4 py-3 text-sm text-right font-mono text-amber-400 font-bold">{m.wickets > 0 ? Array(m.wickets).fill('💥').join(' ') : '-'}</td>
                                          </tr>
                                       ))}
                                    </tbody>
                                 </table>
                              </div>
                           </div>
                        )}
                      </div>
                   )
                ) : playerStats.length === 0 ? (
                   <div className="h-64 flex items-center justify-center text-slate-500 flex-col gap-4">
                     <Activity className="w-8 h-8 animate-spin text-cyan-500" />
                     <p className="font-medium tracking-wide">Loading historical matrices...</p>
                   </div>
                ) : (() => {
                   const isBat = modalPlayer.role === 'BAT' || modalPlayer.role === 'WK';
                   const isBowl = modalPlayer.role === 'BOWL';
                   const isAr = modalPlayer.role === 'AR';

                   return (
                     <>
                       {/* Stats Row */}
                       <div className={cn("grid gap-4 shrink-0", isAr ? "grid-cols-2 md:grid-cols-4" : "grid-cols-1 md:grid-cols-3")}>
                          <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl flex flex-col items-center justify-center shadow-inner">
                            <span className="text-[10px] sm:text-xs text-slate-500 uppercase font-black tracking-widest mb-1.5 text-center">Avg Points (L5)</span>
                            <span className="text-2xl sm:text-3xl font-black text-emerald-400 font-mono drop-shadow-md">{(playerStats.reduce((sum, s) => sum + s.fantasy_points, 0) / playerStats.length).toFixed(1)}</span>
                          </div>
                          
                          {(isBat || isAr) && (
                             <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl flex flex-col items-center justify-center shadow-inner">
                               <span className="text-[10px] sm:text-xs text-slate-500 uppercase font-black tracking-widest mb-1.5 text-center">Avg Strike Rate</span>
                               <span className="text-2xl sm:text-3xl font-black text-cyan-400 font-mono drop-shadow-md">{(playerStats.reduce((sum, s) => sum + s.strike_rate, 0) / playerStats.length).toFixed(1)}</span>
                             </div>
                          )}

                          {(isBowl || isAr) && (
                             <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl flex flex-col items-center justify-center shadow-inner">
                               <span className="text-[10px] sm:text-xs text-slate-500 uppercase font-black tracking-widest mb-1.5 text-center">Total Wickets (L5)</span>
                               <span className="text-2xl sm:text-3xl font-black text-amber-400 font-mono drop-shadow-md">{playerStats.reduce((sum, s) => sum + s.wickets, 0)}</span>
                             </div>
                          )}

                          {isBowl && (
                             <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl flex flex-col items-center justify-center shadow-inner">
                               <span className="text-[10px] sm:text-xs text-slate-500 uppercase font-black tracking-widest mb-1.5 text-center">Avg Economy</span>
                               <span className="text-2xl sm:text-3xl font-black text-rose-400 font-mono drop-shadow-md">{(playerStats.reduce((sum, s) => sum + s.economy, 0) / playerStats.length).toFixed(1)}</span>
                             </div>
                          )}
                          
                          {isBat && (
                             <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl flex flex-col items-center justify-center shadow-inner">
                               <span className="text-[10px] sm:text-xs text-slate-500 uppercase font-black tracking-widest mb-1.5 text-center">Total Runs (L5)</span>
                               <span className="text-2xl sm:text-3xl font-black text-rose-400 font-mono drop-shadow-md">{playerStats.reduce((sum, s) => sum + s.runs_scored, 0)}</span>
                             </div>
                          )}
                          
                          {isAr && (
                             <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl flex flex-col items-center justify-center shadow-inner">
                               <span className="text-[10px] sm:text-xs text-slate-500 uppercase font-black tracking-widest mb-1.5 text-center">Avg Economy</span>
                               <span className="text-2xl sm:text-3xl font-black text-rose-400 font-mono drop-shadow-md">{(playerStats.reduce((sum, s) => sum + s.economy, 0) / playerStats.length).toFixed(1)}</span>
                             </div>
                          )}
                       </div>

                       {/* Charts Row */}
                       <div className={cn("grid gap-6 shrink-0", isAr ? "grid-cols-1 lg:grid-cols-3" : "grid-cols-1 md:grid-cols-2")}>
                         
                         {/* Common Form Chart */}
                         <div className="h-72 bg-slate-900 rounded-xl border border-slate-800 p-5 shadow-lg w-full">
                            <h3 className="text-sm font-bold text-slate-300 mb-6 flex items-center gap-2"><Activity className="w-4 h-4 text-emerald-400" /> Recent Form (Fantasy Pts)</h3>
                            <ResponsiveContainer width="100%" height="85%">
                              <BarChart data={playerStats}>
                                 <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                                 <XAxis dataKey="opponent" tick={{ fill: '#64748b', fontSize: 11, fontWeight: 600 }} tickLine={false} axisLine={false} />
                                 <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} width={30} />
                                 <Tooltip cursor={{ fill: '#0f172a' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff' }} itemStyle={{ color: '#10b981', fontWeight: 'bold' }} />
                                 <Bar dataKey="fantasy_points" fill="#10b981" radius={[4, 4, 0, 0]} maxBarSize={40} />
                              </BarChart>
                            </ResponsiveContainer>
                         </div>

                         {/* Batting Chart */}
                         {(isBat || isAr) && (
                           <div className="h-72 bg-slate-900 rounded-xl border border-slate-800 p-5 shadow-lg w-full">
                              <h3 className="text-sm font-bold text-slate-300 mb-6 flex items-center gap-2"><Target className="w-4 h-4 text-cyan-400" /> Runs Scored Trajectory</h3>
                              <ResponsiveContainer width="100%" height="85%">
                                <BarChart data={playerStats}>
                                   <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                                   <XAxis dataKey="opponent" tick={{ fill: '#64748b', fontSize: 11, fontWeight: 600 }} tickLine={false} axisLine={false} />
                                   <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} width={30} />
                                   <Tooltip cursor={{ fill: '#0f172a' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff' }} />
                                   <Legend wrapperStyle={{ fontSize: '11px', fontWeight: 600, paddingTop: '10px' }} />
                                   <Bar dataKey="non_boundary_runs" stackId="a" fill="#3b82f6" name="Running" maxBarSize={40} />
                                   <Bar dataKey="boundary_runs" stackId="a" fill="#06b6d4" name="Boundaries" radius={[4, 4, 0, 0]} maxBarSize={40} />
                                </BarChart>
                              </ResponsiveContainer>
                           </div>
                         )}

                         {/* Bowling Chart */}
                         {(isBowl || isAr) && (
                           <div className="h-72 bg-slate-900 rounded-xl border border-slate-800 p-5 shadow-lg w-full">
                              <h3 className="text-sm font-bold text-slate-300 mb-6 flex items-center gap-2"><Target className="w-4 h-4 text-amber-400" /> Wickets Taken Trajectory</h3>
                              <ResponsiveContainer width="100%" height="85%">
                                <BarChart data={playerStats}>
                                   <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                                   <XAxis dataKey="opponent" tick={{ fill: '#64748b', fontSize: 11, fontWeight: 600 }} tickLine={false} axisLine={false} />
                                   <YAxis tickFormatter={value => (Number.isInteger(value) ? value : '')} allowDecimals={false} tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} width={30} />
                                   <Tooltip cursor={{ fill: '#0f172a' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff' }} itemStyle={{ color: '#fbbf24', fontWeight: 'bold' }} />
                                   <Bar dataKey="wickets" fill="#fbbf24" name="Wickets" radius={[4, 4, 0, 0]} maxBarSize={40} />
                                </BarChart>
                              </ResponsiveContainer>
                           </div>
                         )}

                       </div>
                     </>
                   );
                })()}
             </div>
          </div>
        </div>
      )}

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #0f172a; 
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #334155; 
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #475569; 
        }
        
        @keyframes scan {
          0% { top: 0%; box-shadow: 0 -20px 20px rgba(34,211,238,0.2) }
          50% { top: 100%; box-shadow: 0 20px 20px rgba(34,211,238,0.2) }
          100% { top: 0%; box-shadow: 0 -20px 20px rgba(34,211,238,0.2) }
        }
        .animate-scan {
          animation: scan 2s linear infinite;
        }
      `}</style>
    </div>
  );
}

function PlayerNode({ player, top, left, pos = "absolute", onSwapClick, swapSourceActive }) {
  if (!player) return null;
  
  const style = pos === 'absolute' ? { top, left, transform: 'translate(-50%, -50%)' } : {};
  
  return (
    <div 
      className={cn(
        "flex flex-col items-center justify-center transition-all duration-500",
        pos === 'absolute' && "absolute",
        onSwapClick ? "cursor-pointer" : ""
      )}
      style={style}
      onClick={() => onSwapClick && onSwapClick(player)}
    >
      <div className={cn(
        "relative group transition-all duration-300",
        !swapSourceActive && "hover:scale-110",
        swapSourceActive && "scale-125 z-20"
      )}>
        <div className={cn(
          "w-12 h-12 bg-slate-900 rounded-full border-2 flex items-center justify-center z-10 relative overflow-hidden transition-all duration-300",
          !swapSourceActive ? "border-emerald-400 shadow-[0_0_15px_rgba(52,211,153,0.3)] hover:shadow-[0_0_20px_rgba(52,211,153,0.5)]" : "border-cyan-400 shadow-[0_0_25px_rgba(34,211,238,0.8)]"
        )}>
           <User className={cn("w-6 h-6 absolute transition-opacity duration-300", swapSourceActive ? "text-cyan-400 opacity-80" : "text-slate-400 opacity-50")} />
           <span className="font-bold text-white text-[10px] z-10 bg-slate-900/80 px-1 rounded truncate max-w-[40px]">{player.name.split(' ')[1] || player.name}</span>
        </div>
        
        {/* Tooltip */}
        <div className="absolute top-full mt-2 left-1/2 -translate-x-1/2 bg-slate-900 border border-slate-700 rounded p-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 whitespace-nowrap shadow-xl">
          <div className="text-xs font-bold text-white flex items-center gap-1.5">{player.name} {player.form_tag === 'hot' && '🔥'}</div>
          <div className="text-[10px] text-emerald-400">{player.proj_points.toFixed(1)} pts</div>
        </div>
        {player.isCaptain && (
          <div className="absolute -top-2 -right-2 bg-amber-500 text-slate-900 text-[9px] font-black w-4 h-4 rounded-full flex items-center justify-center border-2 border-slate-950 z-20">
            C
          </div>
        )}
        {player.isViceCaptain && (
          <div className="absolute -top-2 -right-2 bg-slate-200 text-slate-900 text-[9px] font-black w-4 h-4 rounded-full flex items-center justify-center border-2 border-slate-950 z-20">
            VC
          </div>
        )}
      </div>
      <div className={cn(
        "mt-1 backdrop-blur text-[10px] font-black tracking-wider px-2 py-0.5 rounded border transition-colors",
        swapSourceActive ? "bg-cyan-950/80 text-cyan-400 border-cyan-400/50" : "bg-slate-950/80 text-emerald-400 border-emerald-400/30"
      )}>
        {player.role}
      </div>
    </div>
  );
}

function PlaceholderNode({ role }) {
  return (
    <div className="w-12 h-12 rounded-full border-2 border-dashed border-white/30 flex items-center justify-center bg-black/20 backdrop-blur-sm">
      <span className="text-[10px] font-bold text-white/50">{role}</span>
    </div>
  );
}
