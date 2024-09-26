import { ss } from '@/utils/storage'

const LOCAL_NAME = 'settingsStorage'

export interface SettingsState {
  // systemMessage: string
  temperature: number | null
  top_p: number | null
  top_k: number | null
  // ... other settings
}

export function defaultSetting(): SettingsState {
  return {
    temperature: null,
    top_p: null,
    top_k: null,
    // ... other default settings
  }
}

export function getLocalState(): SettingsState {
  const localSetting: SettingsState | undefined = ss.get(LOCAL_NAME)
  return { ...defaultSetting(), ...localSetting }
}

export function setLocalState(setting: SettingsState): void {
  ss.set(LOCAL_NAME, setting)
}

export function removeLocalState() {
  ss.remove(LOCAL_NAME)
}
