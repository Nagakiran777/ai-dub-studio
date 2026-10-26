export default {
  content: ['./index.html','./src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: { sans: ['Inter','system-ui','sans-serif'] },
      colors: {
        bg:      '#080808',
        surface: '#111111',
        panel:   '#161616',
        card:    '#1c1c1c',
        border:  '#242424',
        border2: '#2e2e2e',
        accent:  '#f5a623',
        accentd: '#c47d0e',
        blue:    '#3b82f6',
        green:   '#22c55e',
        red:     '#ef4444',
        purple:  '#a855f7',
      },
      backdropBlur: { xs: '2px' },
      boxShadow: {
        glow:   '0 0 20px rgba(245,166,35,0.15)',
        glowsm: '0 0 10px rgba(245,166,35,0.1)',
        card:   '0 4px 24px rgba(0,0,0,0.6)',
        panel:  '0 2px 12px rgba(0,0,0,0.4)',
      }
    }
  },
  plugins: []
}