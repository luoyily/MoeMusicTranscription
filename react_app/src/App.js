import * as React from 'react';
import { createBrowserRouter, RouterProvider, } from "react-router-dom";

import AppBar from '@mui/material/AppBar';
import Link from '@mui/material/Link';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import InboxIcon from '@mui/icons-material/Inbox';
import SettingsIcon from '@mui/icons-material/Settings';
import { Button, Divider, Avatar } from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import PianoIcon from '@mui/icons-material/Piano';

import PianoTrans from './PianoTrans';
import VocalTrans from './VocalTrans';
// import sena from '../public/sena.png'

const drawerWidth = 240;
const theme = createTheme({
  // typography: {
  //   color: '#626567',
  // },
  palette: {
    primary: {
      main: '#B0DAFF',
      light: '#EEFAFF'
    },
    contentBgColor: '#F2F4F7',
    globalTextColor: '#393E46',
    sideBarIconColor: { main: '#B0DAFF' }
  },
  components: {
    MuiListItemButton: {
      styleOverrides: {
        root: ({ ownerState, theme }) => ({
          ":hover": {
            backgroundColor: theme.palette.primary.light,
            // color: theme.palette.primary.dark,

            ".MuiListItemIcon-root": {
              // color: theme.palette.primary.light
            }
          },
          "&.Mui-selected": {
            "&:hover": {
              backgroundColor: theme.palette.primary.light,
            },
            backgroundColor: theme.palette.primary.light,
            // color: theme.palette.primary.main,

            ".MuiListItemIcon-root": {
              // color: theme.palette.primary.light,
            }
          },
          borderRadius: 12
        })
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: ({ ownerState, theme }) => ({
          position: 'static',
          backgroundColor: '#ffffff',
          boxShadow: 'none'
        })
      }
    },
    MuiDrawer: {
      styleOverrides: {
        root: ({ ownerState, theme }) => ({
          width: drawerWidth,
          flexShrink: 0,
          '.MuiDrawer-paper': {
            borderRight: 0,
            width: drawerWidth,
            height: `calc(-64px + 100vh)`,
            boxSizing: 'border-box',
            position: 'relative',
            padding: 16
          },
        })
      }
    },
    MuiTypography: {
      styleOverrides: {
        root: ({ ownerState, theme }) => ({
          color: theme.palette.globalTextColor
        })
      }
    },
    MuiDivider: {
      styleOverrides: {
        root: ({ ownerState, theme }) => ({
          margin: '8px 0px'
        })
      }
    }
  }
});

function MyAppBar() {
  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar      >
        <Toolbar>
          <Avatar alt="icon" src="/assets/sena.png" />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, marginLeft: 4 }}>
            MoeMusicTranscription
          </Typography>
          <Button
            variant='text'
            color='primary'
          ><SettingsIcon></SettingsIcon></Button>


        </Toolbar>
      </AppBar>
    </Box>
  );
}

function SideBar() {

  const [selectedIndex, setSelectedIndex] = React.useState(0);

  const handleListItemClick = (event, index) => {
    setSelectedIndex(index);
  };
  return (<Drawer
    variant='permanent'
    anchor="left"
  >
    <Typography>
      Transcriber
    </Typography>
    <List>
      {/* <Link href="/" underline="none"> */}
        <ListItemButton
          selected={selectedIndex === 0}
          onClick={(event) => handleListItemClick(event, 0)}
        >
          <ListItemIcon>
            <PianoIcon color='sideBarIconColor' />
          </ListItemIcon>
          <ListItemText primary="Piano" />
        </ListItemButton>
      {/* </Link> */}
      {/* <Link href="/vocalTrans" underline="none"> */}
        <ListItemButton
          selected={selectedIndex === 1}
          onClick={(event) => handleListItemClick(event, 1)}
        >
          <ListItemIcon>
            <InboxIcon />
          </ListItemIcon>
          <ListItemText primary="Vocal" />
        </ListItemButton>
      {/* </Link> */}

    </List>
    <Divider></Divider>
    <Typography>Tools</Typography>
    <List>
      <ListItemButton
        selected={selectedIndex === 2}
        onClick={(event) => handleListItemClick(event, 2)}
      >
        <ListItemIcon>
          <InboxIcon />
        </ListItemIcon>
        <ListItemText primary="Quantizer" />
      </ListItemButton>
      <ListItemButton
        selected={selectedIndex === 3}
        onClick={(event) => handleListItemClick(event, 3)}
      >
        <ListItemIcon>
          <InboxIcon />
        </ListItemIcon>
        <ListItemText primary="Melody Extract" />
      </ListItemButton>
    </List>
    <Divider />
    <Typography>Other</Typography>
    <List>
      <ListItemButton
        selected={selectedIndex === 4}
        onClick={(event) => handleListItemClick(event, 4)}
      >
        <ListItemIcon>
          <InboxIcon />
        </ListItemIcon>
        <ListItemText primary="Settings" />
      </ListItemButton>
    </List>
    <Button
      sx={{ color: theme.palette.globalTextColor, fontSize: 12 }}
    >
      v0.0.1 dev
    </Button>
  </Drawer>)

}
function Layout({ children }) {
  return (<Box sx={{ display: 'flex', }}>
    <SideBar></SideBar>
    <Box
      sx={{ 'background-color': theme.palette.contentBgColor, flexGrow: 1, borderRadius: 4, padding: 2, display: 'flex' }}
    >{children}</Box>
  </Box>)


}
function App() {
  const router = createBrowserRouter([
    {
      path: "/",
      element: <PianoTrans></PianoTrans>,
    },
    {
      path: "/vocalTrans",
      element: <VocalTrans></VocalTrans>,
    },
  ]);
  return (<ThemeProvider theme={theme}>
    <Box>
      <MyAppBar></MyAppBar>
      <Layout>
        <RouterProvider router={router} />
      </Layout>
    </Box>
  </ThemeProvider>
  )
}
export default App;
