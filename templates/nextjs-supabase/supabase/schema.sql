-- Enable Row Level Security
-- This is a starter schema. Customize tables for your project.

-- Profiles table (extends Supabase auth.users)
create table if not exists public.profiles (
  id uuid references auth.users on delete cascade primary key,
  email text,
  full_name text,
  avatar_url text,
  stripe_customer_id text,
  subscription_status text default 'inactive',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

alter table public.profiles enable row level security;

create policy "Users can view own profile"
  on public.profiles for select
  using (auth.uid() = id);

create policy "Users can update own profile"
  on public.profiles for update
  using (auth.uid() = id);

-- Auto-create profile on signup
create or replace function public.handle_new_user()
returns trigger as $$
begin
  insert into public.profiles (id, email, full_name)
  values (
    new.id,
    new.email,
    new.raw_user_meta_data->>'full_name'
  );
  return new;
end;
$$ language plpgsql security definer;

create or replace trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- Example: Items table (replace with your domain model)
-- create table if not exists public.items (
--   id uuid default gen_random_uuid() primary key,
--   user_id uuid references public.profiles(id) on delete cascade not null,
--   title text not null,
--   description text,
--   status text default 'active',
--   created_at timestamptz default now(),
--   updated_at timestamptz default now()
-- );
--
-- alter table public.items enable row level security;
--
-- create policy "Users can CRUD own items"
--   on public.items for all
--   using (auth.uid() = user_id);
